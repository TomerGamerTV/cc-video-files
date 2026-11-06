-- player.lua
-- Universal CC:Tweaked Media Player (Videos with Audio, Audio-Only MP3/WAV/DFPWM, Animated GIFs, and Static Images)
-- Plays `.ccv` and `.dfpwm` files hosted on GitHub (or any Raw HTTP URL) or local disk.
--
-- Key Features:
-- 1. Automatic HTTP Range Chunked Streaming ("Response is too large" Bypass):
--    CC:Tweaked limits single HTTP responses to `max_download` (16 MiB by default).
--    `player.lua` bypasses this completely by streaming remote URLs in 4 MiB byte-range
--    windows (`Range: bytes=start-end`), allowing arbitrarily long videos/music to play!
-- 2. Automatic Monitor Scaling:
--    Detects any attached Monitor, sets `setTextScale(0.5)` for maximum pixel density,
--    and dynamically scales the video/GIF/image subpixel grid in real time to fit your
--    exact monitor dimensions (`char_w * 2` x `char_h * 3` subpixels)!
-- 3. Full Audio-Only & Video+Audio Support:
--    - Plays `.ccv` Audio-Only streams (`flags` bit 3) with a live visualizer on the monitor/terminal.
--    - Also directly plays raw `.dfpwm` audio URLs or local files out of the box!
-- 4. Automatic Looping for Videos, Audio, and GIFs:
--    Automatically loops continuously (pass `--once` to play only once).
--
-- Usage on PebbleHost / CC:Tweaked:
--   player <github_url_or_local_file> [--once]

local args = { ... }
if #args < 1 then
    print("Usage:")
    print("  player <github_url_or_file> [--once]")
    print("Supported formats:")
    print("  - .ccv (Video+Audio, Audio-Only, Animated GIF, Static Image)")
    print("  - .dfpwm (Raw DFPWM1a Audio)")
    return
end

local source = args[1]
local play_once = (args[2] == "--once")

-- Normalize any GitHub URL (/blob/ or /raw/refs/heads/ or /raw/) into direct `raw.githubusercontent.com`
local function normalize_github_url(url)
    url = string.gsub(url, "^https?://github%.com/([^/]+)/([^/]+)/blob/", "https://raw.githubusercontent.com/%1/%2/")
    url = string.gsub(url, "^https?://github%.com/([^/]+)/([^/]+)/raw/refs/heads/", "https://raw.githubusercontent.com/%1/%2/")
    url = string.gsub(url, "^https?://github%.com/([^/]+)/([^/]+)/raw/", "https://raw.githubusercontent.com/%1/%2/")
    return url
end

source = normalize_github_url(source)

-- 1. Locate Display & Configure Maximum Pixel Density (0.5x Text Scale)
local mon = peripheral.find("monitor")
local display = mon or term.current()

if mon then
    mon.setTextScale(0.5)
end

local dst_char_w, dst_char_h = display.getSize()
local dst_sub_w = dst_char_w * 2
local dst_sub_h = dst_char_h * 3

-- 2. Locate Attached Speakers & DFPWM Decoder
local speakers = { peripheral.find("speaker") }
local has_dfpwm, dfpwm = pcall(require, "cc.audio.dfpwm")

print(string.format("Display: %s (%dx%d chars = %dx%d subpixels)",
    mon and "Monitor (0.5x)" or "Terminal", dst_char_w, dst_char_h, dst_sub_w, dst_sub_h))
if #speakers > 0 then
    print(string.format("Speakers attached: %d", #speakers))
end

-- Save original palette so we can restore it cleanly if terminated
local orig_palette = {}
for i = 0, 15 do
    local r, g, b = display.getPaletteColor(2 ^ i)
    orig_palette[i] = { r, g, b }
end

local function restore_display()
    for _, spk in ipairs(speakers) do
        if spk.stop then pcall(spk.stop) end
    end
    for i = 0, 15 do
        if orig_palette[i] then
            display.setPaletteColor(2 ^ i, orig_palette[i][1], orig_palette[i][2], orig_palette[i][3])
        end
    end
    display.setBackgroundColor(colors.black)
    display.setTextColor(colors.white)
    display.clear()
    display.setCursorPos(1, 1)
end

-- Precompute hex lookup and byte lookup tables for fast Lua rendering
local HEX_CHAR = {
    [0] = "0", [1] = "1", [2] = "2", [3] = "3",
    [4] = "4", [5] = "5", [6] = "6", [7] = "7",
    [8] = "8", [9] = "9", [10] = "a", [11] = "b",
    [12] = "c", [13] = "d", [14] = "e", [15] = "f",
}

local TELETEXT_CHAR = {}
for code = 128, 159 do
    TELETEXT_CHAR[code - 128] = string.char(code)
end

local HIGH_NIBBLE = {}
local LOW_NIBBLE = {}
for b = 0, 255 do
    HIGH_NIBBLE[b] = math.floor(b / 16)
    LOW_NIBBLE[b] = b % 16
end

local str_byte = string.byte
local str_sub = string.sub
local tbl_concat = table.concat
local math_floor = math.floor

-- =============================================================================
-- Chunked HTTP Byte-Range Reader (Bypasses CC:Tweaked "Response is too large"!)
-- Fetches remote files in 4 MiB slices (`Range: bytes=start-end`) seamlessly.
-- =============================================================================
local RANGE_CHUNK_SIZE = 4 * 1024 * 1024 -- 4 MiB per HTTP Range request (well below 16 MiB limit)

local function open_binary_stream(path_or_url)
    if not string.match(path_or_url, "^https?://") then
        local f, err = fs.open(path_or_url, "rb")
        if not f then
            error("Could not open local file " .. path_or_url .. ": " .. tostring(err))
        end
        return f
    end

    -- State for chunked HTTP Range reader
    local current_offset = 0
    local total_size = nil
    local buffer = ""
    local buf_pos = 1
    local eof = false

    local function fetch_next_chunk()
        if eof then return false end
        if total_size and current_offset >= total_size then
            eof = true
            return false
        end

        local range_end = current_offset + RANGE_CHUNK_SIZE - 1
        if total_size and range_end >= total_size then
            range_end = total_size - 1
        end

        local headers = {
            ["Range"] = string.format("bytes=%d-%d", current_offset, range_end),
            ["Cache-Control"] = "no-cache",
        }

        local resp, err = http.get(path_or_url, headers, true)
        if not resp then
            -- Fallback: if server doesn't support Range and file is small, try plain http.get on offset 0
            if current_offset == 0 then
                resp, err = http.get(path_or_url, nil, true)
            end
            if not resp then
                error("HTTP request failed for " .. path_or_url .. ": " .. tostring(err))
            end
        end

        local resp_headers = resp.getResponseHeaders and resp.getResponseHeaders() or {}
        for k, v in pairs(resp_headers) do
            if string.lower(k) == "content-range" then
                local total_str = string.match(v, "/(%d+)")
                if total_str then
                    total_size = tonumber(total_str)
                end
            end
        end

        local chunk_data = resp.readAll()
        resp.close()

        if not chunk_data or #chunk_data == 0 then
            eof = true
            return false
        end

        current_offset = current_offset + #chunk_data
        if total_size and current_offset >= total_size then
            eof = true
        elseif #chunk_data < RANGE_CHUNK_SIZE and not total_size then
            -- Server returned entire remaining body
            eof = true
        end

        buffer = chunk_data
        buf_pos = 1
        return true
    end

    local reader = {}

    function reader.read(n)
        n = n or 1
        local out = {}
        local needed = n

        while needed > 0 do
            local avail = #buffer - buf_pos + 1
            if avail <= 0 then
                if not fetch_next_chunk() then
                    break
                end
                avail = #buffer - buf_pos + 1
            end

            local take = (needed < avail) and needed or avail
            out[#out + 1] = str_sub(buffer, buf_pos, buf_pos + take - 1)
            buf_pos = buf_pos + take
            needed = needed - take
        end

        if #out == 0 then
            return nil
        end
        return tbl_concat(out)
    end

    function reader.close()
        buffer = ""
        eof = true
    end

    return reader
end

-- Apply 48-byte RGB palette (16 colors x 3 bytes)
local function apply_palette(palette_bytes, RGB_FLAT)
    for i = 0, 15 do
        local idx = i * 3 + 1
        local r, g, b = str_byte(palette_bytes, idx, idx + 2)
        RGB_FLAT[i * 3 + 1] = r
        RGB_FLAT[i * 3 + 2] = g
        RGB_FLAT[i * 3 + 3] = b
        display.setPaletteColor(2 ^ i, r / 255.0, g / 255.0, b / 255.0)
    end
end

-- Build nearest-neighbor resampling lookup tables from source `(src_sub_w x src_sub_h)`
-- to current monitor `(dst_sub_w x dst_sub_h)` so scaling takes zero math per frame!
local function build_Monitor_Scaler(src_sub_w, src_sub_h, char_w, char_h)
    local target_sub_w = char_w * 2
    local target_sub_h = char_h * 3
    local bytes_per_src_row = math_floor(src_sub_w / 2)

    local x_byte_offset = {}
    local x_is_low = {}
    for dx = 1, target_sub_w do
        local sx = math_floor((dx - 0.5) * src_sub_w / target_sub_w)
        if sx < 0 then sx = 0 elseif sx >= src_sub_w then sx = src_sub_w - 1 end
        x_byte_offset[dx] = math_floor(sx / 2)
        x_is_low[dx] = (sx % 2) == 1
    end

    local y_row_base = {}
    for dy = 1, target_sub_h do
        local sy = math_floor((dy - 0.5) * src_sub_h / target_sub_h)
        if sy < 0 then sy = 0 elseif sy >= src_sub_h then sy = src_sub_h - 1 end
        y_row_base[dy] = sy * bytes_per_src_row + 1
    end

    return {
        x_byte_offset = x_byte_offset,
        x_is_low = x_is_low,
        y_row_base = y_row_base,
    }
end

-- Scale the 4-bit subpixel buffer to the monitor's exact resolution and pack into 2x3 teletext chars
local function render_scaled_ccv2_frame(packed_bytes, RGB_FLAT, scaler, char_w, char_h)
    local x_byte_offset = scaler.x_byte_offset
    local x_is_low = scaler.x_is_low
    local y_row_base = scaler.y_row_base

    local text_chars = {}
    local fg_chars = {}
    local bg_chars = {}

    for cy = 1, char_h do
        local dy0 = (cy - 1) * 3 + 1
        local row_base0 = y_row_base[dy0]
        local row_base1 = y_row_base[dy0 + 1]
        local row_base2 = y_row_base[dy0 + 2]

        for cx = 1, char_w do
            local dx0 = (cx - 1) * 2 + 1
            local dx1 = dx0 + 1

            local off0 = x_byte_offset[dx0]
            local low0 = x_is_low[dx0]
            local off1 = x_byte_offset[dx1]
            local low1 = x_is_low[dx1]

            local b00 = str_byte(packed_bytes, row_base0 + off0)
            local b01 = (off0 == off1) and b00 or str_byte(packed_bytes, row_base0 + off1)
            local c0 = low0 and LOW_NIBBLE[b00] or HIGH_NIBBLE[b00]
            local c1 = low1 and LOW_NIBBLE[b01] or HIGH_NIBBLE[b01]

            local b10 = str_byte(packed_bytes, row_base1 + off0)
            local b11 = (off0 == off1) and b10 or str_byte(packed_bytes, row_base1 + off1)
            local c2 = low0 and LOW_NIBBLE[b10] or HIGH_NIBBLE[b10]
            local c3 = low1 and LOW_NIBBLE[b11] or HIGH_NIBBLE[b11]

            local b20 = str_byte(packed_bytes, row_base2 + off0)
            local b21 = (off0 == off1) and b20 or str_byte(packed_bytes, row_base2 + off1)
            local c4 = low0 and LOW_NIBBLE[b20] or HIGH_NIBBLE[b20]
            local c5 = low1 and LOW_NIBBLE[b21] or HIGH_NIBBLE[b21]

            local bg = c5
            local fg = c5
            if c0 ~= bg then
                fg = c0
            elseif c1 ~= bg then
                fg = c1
            elseif c2 ~= bg then
                fg = c2
            elseif c3 ~= bg then
                fg = c3
            elseif c4 ~= bg then
                fg = c4
            end

            if fg == bg then
                text_chars[cx] = TELETEXT_CHAR[0]
                fg_chars[cx] = HEX_CHAR[bg]
                bg_chars[cx] = HEX_CHAR[bg]
            else
                local f_base = fg * 3
                local b_base = bg * 3
                local fr, fg_g, fb = RGB_FLAT[f_base + 1], RGB_FLAT[f_base + 2], RGB_FLAT[f_base + 3]
                local br, bg_g, bb = RGB_FLAT[b_base + 1], RGB_FLAT[b_base + 2], RGB_FLAT[b_base + 3]

                local code = 0

                if c0 == fg then
                    code = code + 1
                elseif c0 ~= bg then
                    local p = c0 * 3
                    local r, g, b = RGB_FLAT[p + 1], RGB_FLAT[p + 2], RGB_FLAT[p + 3]
                    local df = (r - fr) * (r - fr) + (g - fg_g) * (g - fg_g) + (b - fb) * (b - fb)
                    local db = (r - br) * (r - br) + (g - bg_g) * (g - bg_g) + (b - bb) * (b - bb)
                    if df < db then code = code + 1 end
                end

                if c1 == fg then
                    code = code + 2
                elseif c1 ~= bg then
                    local p = c1 * 3
                    local r, g, b = RGB_FLAT[p + 1], RGB_FLAT[p + 2], RGB_FLAT[p + 3]
                    local df = (r - fr) * (r - fr) + (g - fg_g) * (g - fg_g) + (b - fb) * (b - fb)
                    local db = (r - br) * (r - br) + (g - bg_g) * (g - bg_g) + (b - bb) * (b - bb)
                    if df < db then code = code + 2 end
                end

                if c2 == fg then
                    code = code + 4
                elseif c2 ~= bg then
                    local p = c2 * 3
                    local r, g, b = RGB_FLAT[p + 1], RGB_FLAT[p + 2], RGB_FLAT[p + 3]
                    local df = (r - fr) * (r - fr) + (g - fg_g) * (g - fg_g) + (b - fb) * (b - fb)
                    local db = (r - br) * (r - br) + (g - bg_g) * (g - bg_g) + (b - bb) * (b - bb)
                    if df < db then code = code + 4 end
                end

                if c3 == fg then
                    code = code + 8
                elseif c3 ~= bg then
                    local p = c3 * 3
                    local r, g, b = RGB_FLAT[p + 1], RGB_FLAT[p + 2], RGB_FLAT[p + 3]
                    local df = (r - fr) * (r - fr) + (g - fg_g) * (g - fg_g) + (b - fb) * (b - fb)
                    local db = (r - br) * (r - br) + (g - bg_g) * (g - bg_g) + (b - bb) * (b - bb)
                    if df < db then code = code + 8 end
                end

                if c4 == fg then
                    code = code + 16
                elseif c4 ~= bg then
                    local p = c4 * 3
                    local r, g, b = RGB_FLAT[p + 1], RGB_FLAT[p + 2], RGB_FLAT[p + 3]
                    local df = (r - fr) * (r - fr) + (g - fg_g) * (g - fg_g) + (b - fb) * (b - fb)
                    local db = (r - br) * (r - br) + (g - bg_g) * (g - bg_g) + (b - bb) * (b - bb)
                    if df < db then code = code + 16 end
                end

                text_chars[cx] = TELETEXT_CHAR[code]
                fg_chars[cx] = HEX_CHAR[fg]
                bg_chars[cx] = HEX_CHAR[bg]
            end
        end

        display.setCursorPos(1, cy)
        display.blit(tbl_concat(text_chars), tbl_concat(fg_chars), tbl_concat(bg_chars))
    end
end

-- Simple audio-only visualizer on the display
local function render_audio_visualizer(samples, char_w, char_h, elapsed_sec)
    display.setBackgroundColor(colors.black)
    display.clear()
    local mid_y = math_floor(char_h / 2)
    display.setCursorPos(2, 2)
    display.setTextColor(colors.cyan)
    display.write(string.format("Playing Audio Stream (%.1fs)", elapsed_sec))

    if not samples or #samples == 0 then return end
    local step = math.max(1, math_floor(#samples / char_w))
    for x = 1, char_w do
        local s = samples[(x - 1) * step + 1] or 0
        local amp = math_floor((math.abs(s) / 128.0) * (char_h / 2 - 2))
        local y_top = math.max(3, mid_y - amp)
        local y_bot = math.min(char_h - 1, mid_y + amp)
        for y = y_top, y_bot do
            display.setCursorPos(x, y)
            display.setBackgroundColor((y == mid_y) and colors.lightBlue or colors.blue)
            display.write(" ")
        end
    end
    display.setBackgroundColor(colors.black)
end

-- Play a raw `.dfpwm` file or stream directly
local function play_raw_dfpwm_stream(handle)
    local decoder = (has_dfpwm and dfpwm) and dfpwm.make_decoder() or nil
    if not decoder or #speakers == 0 then
        handle.close()
        error("No speaker or cc.audio.dfpwm available for audio playback.")
    end

    if mon then mon.setTextScale(0.5) end
    local char_w, char_h = display.getSize()
    local elapsed = 0.0
    local chunk_size = 6000 -- 1.0s of 48kHz DFPWM audio (48,000 samples)

    while true do
        local chunk = handle.read(chunk_size)
        if not chunk or #chunk == 0 then break end
        local samples = decoder(chunk)
        render_audio_visualizer(samples, char_w, char_h, elapsed)
        elapsed = elapsed + (#chunk / 6000.0)

        for _, spk in ipairs(speakers) do
            while not spk.playAudio(samples, 3.0) do
                os.pullEvent("speaker_audio_empty")
            end
        end
    end

    handle.close()
    return false
end

local function play_stream_once(path_or_url)
    local handle = open_binary_stream(path_or_url)

    -- If URL/path ends with .dfpwm, play directly as raw DFPWM audio
    if string.match(string.lower(path_or_url), "%.dfpwm$") then
        return play_raw_dfpwm_stream(handle)
    end

    local magic = handle.read(4)
    if magic ~= "CCV2" then
        handle.close()
        error("Unsupported file header (expected 'CCV2', got '" .. tostring(magic) .. "'). Re-convert with convert_and_stream.py.")
    end

    local hdr = handle.read(8)
    local src_sub_w = str_byte(hdr, 1) * 256 + str_byte(hdr, 2)
    local src_sub_h = str_byte(hdr, 3) * 256 + str_byte(hdr, 4)
    local fps = str_byte(hdr, 5)
    local flags = str_byte(hdr, 6)

    local has_audio = (flags % 2) == 1
    local is_static_image = (math_floor(flags / 2) % 2) == 1
    local is_audio_only = (math_floor(flags / 8) % 2) == 1 or (src_sub_w == 0 and src_sub_h == 0)

    -- Fresh DFPWM decoder instance for each playback pass so filter state starts clean
    local decoder = (has_dfpwm and dfpwm) and dfpwm.make_decoder() or nil
    local pending_audio_chunks = {}
    local pending_audio_bytes = 0

    local function flush_audio(force)
        if #speakers == 0 or not decoder or pending_audio_bytes == 0 then
            return
        end
        if not force and pending_audio_bytes < 1200 then
            return
        end
        local combined = tbl_concat(pending_audio_chunks)
        local samples = decoder(combined)
        local accepted = false
        for _, spk in ipairs(speakers) do
            if spk.playAudio(samples, 3.0) then
                accepted = true
            end
        end
        if accepted or pending_audio_bytes >= 3000 then
            pending_audio_chunks = {}
            pending_audio_bytes = 0
        end
        return samples
    end

    if mon then mon.setTextScale(0.5) end
    local char_w, char_h = display.getSize()

    -- Handle CCV2 Audio-Only mode (`is_audio_only`)
    if is_audio_only then
        local elapsed = 0.0
        while true do
            local chunk_type_str = handle.read(1)
            if not chunk_type_str or str_byte(chunk_type_str, 1) == 0xFF then break end
            local alen_str = handle.read(2)
            if not alen_str or #alen_str < 2 then break end
            local audio_len = str_byte(alen_str, 1) * 256 + str_byte(alen_str, 2)
            if audio_len > 0 then
                local audio_data = handle.read(audio_len)
                if not audio_data then break end
                if decoder and #speakers > 0 then
                    local samples = decoder(audio_data)
                    render_audio_visualizer(samples, char_w, char_h, elapsed)
                    elapsed = elapsed + (#audio_data / 6000.0)
                    for _, spk in ipairs(speakers) do
                        while not spk.playAudio(samples, 3.0) do
                            os.pullEvent("speaker_audio_empty")
                        end
                    end
                else
                    sleep(#audio_data / 6000.0)
                end
            end
        end
        handle.close()
        return false
    end

    local scaler = build_Monitor_Scaler(src_sub_w, src_sub_h, char_w, char_h)
    local packed_frame_bytes = math_floor((src_sub_w * src_sub_h) / 2)
    local RGB_FLAT = {}

    local frame_interval = (fps > 0) and (1.0 / fps) or 0.05
    local start_clock = os.epoch("utc") / 1000.0
    local frame_index = 0

    while true do
        local chunk_type_str = handle.read(1)
        if not chunk_type_str then break end
        local chunk_type = str_byte(chunk_type_str, 1)
        if chunk_type == 0xFF then break end

        local alen_str = handle.read(2)
        if not alen_str or #alen_str < 2 then break end
        local audio_len = str_byte(alen_str, 1) * 256 + str_byte(alen_str, 2)
        local audio_data = nil
        if audio_len > 0 then
            audio_data = handle.read(audio_len)
        end

        local palette_data = handle.read(48)
        if not palette_data or #palette_data < 48 then break end

        local packed_px = handle.read(packed_frame_bytes)
        if not packed_px or #packed_px < packed_frame_bytes then break end

        if has_audio and audio_data and #audio_data > 0 then
            pending_audio_chunks[#pending_audio_chunks + 1] = audio_data
            pending_audio_bytes = pending_audio_bytes + #audio_data
            flush_audio(frame_index == 0)
        end

        apply_palette(palette_data, RGB_FLAT)
        render_scaled_ccv2_frame(packed_px, RGB_FLAT, scaler, char_w, char_h)

        if is_static_image then
            break
        end

        frame_index = frame_index + 1
        local target_time = start_clock + (frame_index * frame_interval)
        local now = os.epoch("utc") / 1000.0
        local wait_time = target_time - now
        if wait_time > 0.005 then
            sleep(wait_time)
        else
            os.queueEvent("yield")
            os.pullEvent("yield")
        end
    end

    flush_audio(true)
    handle.close()
    return is_static_image
end

-- Main execution loop: loops all videos, audio, and GIFs continuously unless `--once` is passed!
local ok, err = pcall(function()
    print("Playing (loops automatically; press Ctrl+T to stop)...")
    while true do
        local is_static_image = play_stream_once(source)
        if is_static_image then
            print("Displaying static image on monitor. Press Enter in terminal to clear, or Ctrl+T to keep on monitor.")
            os.pullEvent("key")
            break
        elseif play_once then
            break
        else
            sleep(0.05)
        end
    end
end)

if not ok then
    restore_display()
    printError("Playback stopped: " .. tostring(err))
else
    print("Finished.")
end
