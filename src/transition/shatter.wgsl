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
    grid_x: u32,
    grid_y: u32,
    _pad: u32,
};

@group(0) @binding(0)
var from_tex: texture_2d<f32>;

@group(0) @binding(1)
var to_tex: texture_2d<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

const SMOOTHNESS: f32 = 0.25;
const FOCAL: f32 = 2.0;
const Z_MAX: f32 = 0.3;

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

// How far along `dir` from `pos` before pos + dir*t clears the viewport
// boundary, expanded by `margin` on each side to account for perspective.
fn exit_distance(pos: vec2<f32>, dir: vec2<f32>, margin: f32) -> f32 {
    var t: f32 = 99.0;
    if (dir.x > 0.001) { t = min(t, (1.0 + margin - pos.x) / dir.x); }
    if (dir.x < -0.001) { t = min(t, (-margin - pos.x) / dir.x); }
    if (dir.y > 0.001) { t = min(t, (1.0 + margin - pos.y) / dir.y); }
    if (dir.y < -0.001) { t = min(t, (-margin - pos.y) / dir.y); }
    return t;
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
    return textureLoad(from_tex, vec2<u32>(sx, sy), 0);
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
    return textureLoad(to_tex, vec2<u32>(sx, sy), 0);
}

// Per-cell animation properties derived from the grid position and seed.
struct CellProps {
    center: vec2<f32>,
    exit_dir: vec2<f32>,
    exit_dist: f32,
    max_zoom: f32,
    order: f32,
};

fn compute_cell_props(cx: u32, cy: u32) -> CellProps {
    let grid = vec2<f32>(f32(params.grid_x), f32(params.grid_y));
    let cell_center = (vec2<f32>(f32(cx), f32(cy)) + 0.5) / grid;

    let r_base  = hash01_u32(cx, cy, params.seed_lo, params.seed_hi);
    let r_angle = hash_channel(cx, cy, 1u, params.seed_lo, params.seed_hi);
    let r_speed = hash_channel(cx, cy, 3u, params.seed_lo, params.seed_hi);

    let radial = cell_center - vec2<f32>(0.5, 0.5);
    let radial_dist = length(radial);
    let radial_dir = radial / max(radial_dist, 0.001);
    let jitter_angle = r_angle * 6.283185;
    let jitter_dir = vec2<f32>(cos(jitter_angle), sin(jitter_angle));
    let blend = smoothstep(0.0, 0.15, radial_dist);
    let blended = mix(jitter_dir, radial_dir, blend);
    let blended_len = length(blended);
    let exit_dir = select(jitter_dir, blended / blended_len, blended_len > 0.001);

    let max_zoom = 1.1 + r_speed * 0.15;

    let half_cell = 0.5 / grid;
    let cell_extent = (abs(exit_dir.x) * half_cell.x + abs(exit_dir.y) * half_cell.y) * max_zoom;
    let perspective_margin = 0.5 * Z_MAX / FOCAL;
    let exit_dist = exit_distance(cell_center, exit_dir, perspective_margin) + cell_extent;

    let centre_factor = clamp(radial_dist / 0.707, 0.0, 1.0);
    let order = 0.25 + r_base * 0.25 + (1.0 - centre_factor) * 0.25;

    return CellProps(cell_center, exit_dir, exit_dist, max_zoom, order);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) is_stage2: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOutput {
    let cx = iid % params.grid_x;
    let cy = iid / params.grid_x;
    let grid = vec2<f32>(f32(params.grid_x), f32(params.grid_y));

    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let local = corners[vid];
    let cell_uv = (vec2<f32>(f32(cx), f32(cy)) + local) / grid;

    let props = compute_cell_props(cx, cy);
    let progress = timeline_progress(params.frame_start);

    var motion: f32;
    var stage2: f32;
    if (progress < 0.5) {
        let phase = progress * 2.0;
        motion = cell_progress(props.order, phase);
        stage2 = 0.0;
    } else {
        let phase = (progress - 0.5) * 2.0;
        let order_in = 1.0 - props.order;
        motion = 1.0 - cell_progress(order_in, phase);
        stage2 = 1.0;
    }

    let zoom = 1.0 + motion * (props.max_zoom - 1.0);
    let drift = props.exit_dir * motion * props.exit_dist;
    let displaced = props.center + (cell_uv - props.center) * zoom + drift;

    let z = motion * Z_MAX;
    let scale = FOCAL / (FOCAL + z);
    let screen = vec2<f32>(0.5) + (displaced - vec2<f32>(0.5)) * scale;

    let ndc_x = screen.x * 2.0 - 1.0;
    let ndc_y = 1.0 - screen.y * 2.0;
    let instance_count = max(1.0, f32(params.grid_x) * f32(params.grid_y));
    let depth = 0.5 - motion * 0.45 + f32(iid) / instance_count * 0.02;

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, depth, 1.0);
    out.uv = cell_uv;
    out.is_stage2 = stage2;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;
    if (in.is_stage2 < 0.5) {
        color = fit_sample_from(in.uv);
    } else {
        color = fit_sample_to(in.uv);
    }
    if (color.a < 0.5) {
        discard;
    }
    return color;
}
