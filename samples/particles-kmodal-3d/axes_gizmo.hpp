#pragma once

// Shared geometry for the --axes orientation triad, used by BOTH benchmark_mimir and
// benchmark_datoviz so the two draw byte-identical gizmos: the three POSITIVE half-axes from
// the origin (X=red, Y=green, Z=blue) plus an X/Y/Z letter at each tip.
//
// The axis lines are world-space segments. The letters are BILLBOARDED: every stroke of a
// letter is anchored at the label's single world position and shaped purely by per-endpoint
// SCREEN-SPACE pixel shifts (+x right, +y up), so the letters always face the camera. Both
// renderers apply the shift the same way (datoviz: dvz_segment_shift / transform_shift;
// mimir: the line.slang texcoord shift) -- after projection and not scaled by w, so the
// on-screen size is shift/w pixels: letters keep their orientation but still scale with
// camera distance like world objects. The stroke shifts below are pre-scaled by each label's
// depth from the shared home camera at (0,0,4), so at the home view every letter renders at
// glyph_px pixels.
//
// The gizmo is an unlit, depth-test-off overlay in both benchmarks (flat vertex colors; no
// lighting, no shadowing, no interaction with the particle geometry).

#include <vector>

struct AxisSegment
{
    float ax, ay, az; // start anchor (world)
    float bx, by, bz; // end anchor (world)
    float sax, say;   // screen-space pixel shift at the start vertex (+x right, +y up)
    float sbx, sby;   // screen-space pixel shift at the end vertex
    float r, g, b;    // color
};

inline std::vector<AxisSegment> makeAxesGizmo()
{
    std::vector<AxisSegment> segs;
    const float color[3][3]  = { {1.f,.15f,.15f}, {.15f,1.f,.15f}, {.25f,.45f,1.f} };
    const float axis_len     = 0.75f; // triad stays well inside the [-1,1] domain
    const float label_center = 0.85f; // letter anchor sits just past the axis tip
    const float glyph_px     = 26.f;  // on-screen letter size at the home camera, in pixels
    const float home_dist    = 4.f;   // home camera eye distance (shared by both benchmarks)

    // Three positive half-axes: origin -> +axis_len along each axis (no shifts).
    for (int a = 0; a < 3; ++a)
    {
        AxisSegment s{};
        s.r = color[a][0]; s.g = color[a][1]; s.b = color[a][2];
        if      (a == 0) s.bx = axis_len;
        else if (a == 1) s.by = axis_len;
        else             s.bz = axis_len;
        segs.push_back(s);
    }

    // Letter glyphs as strokes in a unit (u,v) box (v up), anchored at each tip and expressed
    // as pixel shifts around the anchor.
    struct Stroke { float u0, v0, u1, v1; };
    const std::vector<Stroke> letters[3] = {
        /*X*/ { {0,0,1,1}, {0,1,1,0} },
        /*Y*/ { {0,1,.5f,.5f}, {1,1,.5f,.5f}, {.5f,.5f,.5f,0} },
        /*Z*/ { {0,1,1,1}, {1,1,0,0}, {0,0,1,0} },
    };
    const float anchor[3][3] = {
        {label_center,0,0}, {0,label_center,0}, {0,0,label_center},
    };

    for (int a = 0; a < 3; ++a)
    {
        // The renderers divide the pixel shift by the vertex's clip w (= camera distance), so
        // scale each glyph by its label's home-view depth to render glyph_px on screen there.
        // The z label sits closer to the home camera than the x/y ones (its anchor is on the
        // view axis), hence the per-axis factor.
        const float depth = (a == 2) ? home_dist - label_center : home_dist;
        const float scale = glyph_px * depth;
        for (const auto& st : letters[a])
        {
            AxisSegment s{};
            s.r  = color[a][0]; s.g = color[a][1]; s.b = color[a][2];
            s.ax = anchor[a][0]; s.ay = anchor[a][1]; s.az = anchor[a][2];
            s.bx = anchor[a][0]; s.by = anchor[a][1]; s.bz = anchor[a][2];
            s.sax = (st.u0 - .5f) * scale; s.say = (st.v0 - .5f) * scale;
            s.sbx = (st.u1 - .5f) * scale; s.sby = (st.v1 - .5f) * scale;
            segs.push_back(s);
        }
    }
    return segs;
}
