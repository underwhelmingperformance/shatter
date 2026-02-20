struct Params {
    out_width: u32,
    out_height: u32,
    from_width: u32,
    from_height: u32,
    to_width: u32,
    to_height: u32,
    total_frames: u32,
    frame_start: u32,
    chunk_frames: u32,
    hold_frames: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> from_pixels: array<u32>;

@group(0) @binding(1)
var<storage, read> to_pixels: array<u32>;

@group(0) @binding(2)
var<storage, read_write> out_pixels: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

const GRID_X: f32 = 20.0;
const GRID_Y: f32 = 20.0;
const SMOOTHNESS: f32 = 0.25;

fn unpack_rgba(pixel: u32) -> vec4<f32> {
    let r = f32(pixel & 0xFFu) / 255.0;
    let g = f32((pixel >> 8u) & 0xFFu) / 255.0;
    let b = f32((pixel >> 16u) & 0xFFu) / 255.0;
    let a = f32((pixel >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn pack_rgba(pixel: vec4<f32>) -> u32 {
    let clamped = clamp(pixel, vec4<f32>(0.0), vec4<f32>(1.0));
    let r = u32(round(clamped.r * 255.0));
    let g = u32(round(clamped.g * 255.0));
    let b = u32(round(clamped.b * 255.0));
    let a = u32(round(clamped.a * 255.0));
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

fn hash01_u32(x: u32, y: u32, seed_lo: u32, seed_hi: u32) -> f32 {
    var state = x * 374761393u + y * 668265263u + seed_lo * 982451653u + seed_hi * 2654435761u;
    state = (state ^ (state >> 13u)) * 1274126177u;
    state = state ^ (state >> 16u);
    return f32(state & 0x00FFFFFFu) / 16777215.0;
}

fn hash_channel(cx: u32, cy: u32, ch: u32, seed_lo: u32, seed_hi: u32) -> f32 {
    return hash01_u32(cx + ch * 7919u, cy + ch * 6271u, seed_lo, seed_hi);
}

fn cell_progress(cell_noise: f32, phase_progress: f32) -> f32 {
    let edge0 = max(0.0, cell_noise - SMOOTHNESS);
    let edge1 = min(1.0, cell_noise + SMOOTHNESS);
    return smoothstep(edge0, edge1, phase_progress);
}

fn ease_out_quart(x: f32) -> f32 {
    let t = 1.0 - x;
    return 1.0 - t * t * t * t;
}

fn timeline_progress(frame_index: u32) -> f32 {
    if (params.total_frames <= 1u) {
        return 1.0;
    }

    let max_hold = (params.total_frames - 1u) / 2u;
    let hold = min(params.hold_frames, max_hold);
    if (frame_index < hold) {
        return 0.0;
    }

    let final_transition_frame = params.total_frames - hold - 1u;
    if (frame_index >= final_transition_frame) {
        return 1.0;
    }

    let transition_frames = params.total_frames - (hold * 2u);
    if (transition_frames <= 1u) {
        return 1.0;
    }

    let active_index = frame_index - hold;
    return f32(active_index) / f32(transition_frames - 1u);
}

// fit_sample_from and fit_sample_to are near-identical: both do
// aspect-ratio-preserving sampling with letterbox/pillarbox centering.
// WGSL lacks the ability to pass storage buffer references as function
// parameters, so the duplication is unavoidable at the language level.

fn fit_sample_from(uv: vec2<f32>) -> vec4<f32> {
    let out_w = f32(params.out_width);
    let out_h = f32(params.out_height);
    let src_w = f32(params.from_width);
    let src_h = f32(params.from_height);

    let scale = min(out_w / src_w, out_h / src_h);
    let scaled_w = src_w * scale;
    let scaled_h = src_h * scale;
    let offset_x = (out_w - scaled_w) * 0.5;
    let offset_y = (out_h - scaled_h) * 0.5;

    let px = uv.x * out_w;
    let py = uv.y * out_h;

    if (px < offset_x || py < offset_y || px >= offset_x + scaled_w || py >= offset_y + scaled_h) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let u = (px - offset_x) / scaled_w;
    let v = (py - offset_y) / scaled_h;

    let sx = min(u32(floor(u * src_w)), params.from_width - 1u);
    let sy = min(u32(floor(v * src_h)), params.from_height - 1u);
    let index = sy * params.from_width + sx;
    return unpack_rgba(from_pixels[index]);
}

fn fit_sample_to(uv: vec2<f32>) -> vec4<f32> {
    let out_w = f32(params.out_width);
    let out_h = f32(params.out_height);
    let src_w = f32(params.to_width);
    let src_h = f32(params.to_height);

    let scale = min(out_w / src_w, out_h / src_h);
    let scaled_w = src_w * scale;
    let scaled_h = src_h * scale;
    let offset_x = (out_w - scaled_w) * 0.5;
    let offset_y = (out_h - scaled_h) * 0.5;

    let px = uv.x * out_w;
    let py = uv.y * out_h;

    if (px < offset_x || py < offset_y || px >= offset_x + scaled_w || py >= offset_y + scaled_h) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let u = (px - offset_x) / scaled_w;
    let v = (py - offset_y) / scaled_h;

    let sx = min(u32(floor(u * src_w)), params.to_width - 1u);
    let sy = min(u32(floor(v * src_h)), params.to_height - 1u);
    let index = sy * params.to_width + sx;
    return unpack_rgba(to_pixels[index]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.out_width || gid.y >= params.out_height || gid.z >= params.chunk_frames) {
        return;
    }

    let pixel_index = gid.y * params.out_width + gid.x;
    let frame_index = params.frame_start + gid.z;

    let progress = timeline_progress(frame_index);

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(params.out_width),
        (f32(gid.y) + 0.5) / f32(params.out_height)
    );

    let cell = floor(vec2<f32>(GRID_X, GRID_Y) * uv);
    let cx = u32(cell.x);
    let cy = u32(cell.y);
    let cell_center = (cell + 0.5) / vec2<f32>(GRID_X, GRID_Y);

    let r_base  = hash01_u32(cx, cy, params.seed_lo, params.seed_hi);
    let r_angle = hash_channel(cx, cy, 1u, params.seed_lo, params.seed_hi);
    let r_speed = hash_channel(cx, cy, 3u, params.seed_lo, params.seed_hi);

    // Forward speed: how fast this cell flies towards the camera.
    let speed = 0.6 + r_speed * 0.8;

    // Subtle lateral drift direction (random per cell).
    let drift_angle = r_angle * 6.283185;
    let drift_dir = vec2<f32>(cos(drift_angle), sin(drift_angle));

    // Wave ordering: edges crack first, centre holds longest.
    let centre_dist = length(cell_center - vec2<f32>(0.5, 0.5));
    let centre_factor = clamp(centre_dist / 0.707, 0.0, 1.0);
    let order = 0.25 + r_base * 0.25 + (1.0 - centre_factor) * 0.25;

    // Perspective vanishing point.
    let vp = vec2<f32>(0.5, 0.5);

    let out_index = gid.z * (params.out_width * params.out_height) + pixel_index;

    // Stage one: explosion behind the image pushes cells towards the camera.
    if (progress < 0.5) {
        let phase = progress * 2.0;
        let eased = ease_out_quart(phase);

        let disappear = cell_progress(order, phase);

        // Perspective scale: cell approaches camera along Z.
        let z = eased * speed;
        let scale = 1.0 / max(1.0 - z, 0.01);

        // Lateral scatter applied after the perspective division so it stays
        // visible even at high zoom (centre cells won't look static).
        let drift = drift_dir * eased * speed * 0.15;

        let sample_uv = clamp(
            vp + (uv - vp) / scale - drift,
            vec2<f32>(0.0), vec2<f32>(1.0)
        );
        let from_color = fit_sample_from(sample_uv);
        let alpha = from_color.a * (1.0 - disappear) / scale;

        if (alpha <= 0.0) {
            out_pixels[out_index] = pack_rgba(vec4<f32>(0.0, 0.0, 0.0, 0.0));
            return;
        }
        out_pixels[out_index] = pack_rgba(vec4<f32>(from_color.rgb, alpha));
        return;
    }

    // Stage two: target image assembles from scattered state.
    let phase = (progress - 0.5) * 2.0;
    let remaining = 1.0 - phase;

    let appear = cell_progress(order, phase);

    let z = remaining * speed;
    let scale = 1.0 / max(1.0 - z, 0.01);

    let drift = drift_dir * remaining * speed * 0.15;

    let sample_uv = clamp(
        vp + (uv - vp) / scale - drift,
        vec2<f32>(0.0), vec2<f32>(1.0)
    );
    let to_color = fit_sample_to(sample_uv);
    let alpha = to_color.a * appear / scale;

    if (alpha <= 0.0) {
        out_pixels[out_index] = pack_rgba(vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    out_pixels[out_index] = pack_rgba(vec4<f32>(to_color.rgb, alpha));
}
