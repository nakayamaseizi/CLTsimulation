"""
viz_plotly.py
=============
【役割】
Plotly を使ったインタラクティブ可視化モジュール。
Streamlit の「🔬 3D ビューア」タブで使用する。

【提供する関数】
- make_temp_profile_animation() : 温度プロファイルのスライダー付きアニメ
- make_charring_chart()         : 炭化深さの時刻歴（Plotly 版）
- make_surface_temp_chart()     : 加熱面・非加熱面温度の時刻歴（Plotly 版）
- make_char_depth_heatmap()     : 温度場をヒートマップで表示（時間×深さ）
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 温度プロファイル アニメーション
# ---------------------------------------------------------------------------

def make_temp_profile_animation(
    result: dict[str, Any],
    n_frames: int = 40,
    char_temp: float = 300.0,
) -> Any:
    """厚み方向温度プロファイルのインタラクティブアニメーションを作成する。

    スライダーで時刻を選択し、温度プロファイルと炭化前線を表示する。
    カラーは Inferno カラーマップを使用。

    Parameters
    ----------
    result : dict
        solve() の返り値。times, temperatures, x_centers, char_depths を含む。
    n_frames : int
        アニメーションのフレーム数（時刻サンプル数）。
    char_temp : float
        炭化温度 [°C]（炭化前線マーカーに使用）。

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    times_s = result["times"]
    times_min = times_s / 60.0
    x_mm = result["x_centers"] * 1000.0
    T_mat = result["temperatures"]
    char_depths_mm = result["char_depths"] * 1000.0

    # フレームとなる時刻インデックスを均等に選択
    n_total = len(times_s)
    frame_indices = np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)

    # 全フレームの最大温度（カラースケール固定用）
    t_max_global = float(T_mat.max())
    t_max_color = max(t_max_global, 400.0)

    # Inferno カラーマップ風の色変換（時刻の進捗に応じて色が変化）
    def _time_to_color(frac: float) -> str:
        """0–1 の進捗値を Inferno カラーマップの色文字列に変換する。"""
        r = int(255 * (0.8 * frac))
        g = int(255 * (0.2 * frac))
        b = int(255 * (1.0 - frac) * 0.6 + 60)
        return f"rgb({r},{g},{b})"

    # ── 全フレームの traces を準備（常に 2 traces: 温度線 + 炭化前線）──
    frames = []
    for fi in frame_indices:
        t_min = times_min[fi]
        T_profile = T_mat[fi]
        if T_profile.ndim > 1:
            T_profile = T_profile.ravel()[:len(x_mm)]

        char_d = char_depths_mm[fi]
        frac = t_min / float(times_min[-1]) if times_min[-1] > 0 else 0.0
        color = _time_to_color(frac)

        # 温度プロファイル線
        trace_temp = go.Scatter(
            x=x_mm.tolist(),
            y=T_profile.tolist(),
            mode="lines",
            name=f"{t_min:.1f}分",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.08)"),
        )

        # 炭化前線（常に含める；炭化がなければ非表示領域に押し込む）
        if char_d > 0:
            trace_char = go.Scatter(
                x=[char_d, char_d],
                y=[0, char_temp],
                mode="lines",
                name="炭化前線",
                line=dict(color="orange", width=2, dash="dash"),
            )
        else:
            # 炭化未発生：空のトレース（座標は画面外）
            trace_char = go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="炭化前線",
                line=dict(color="orange", width=2, dash="dash"),
            )

        frames.append(go.Frame(
            data=[trace_temp, trace_char],
            name=f"{t_min:.1f}",
        ))

    # ── 初期表示（最初のフレームと同じ）──
    fi0 = frame_indices[0]
    T0 = T_mat[fi0]
    if T0.ndim > 1:
        T0 = T0.ravel()[:len(x_mm)]
    char_d0 = char_depths_mm[fi0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_mm.tolist(),
                y=T0.tolist(),
                mode="lines",
                name="温度分布",
                line=dict(color="rgb(50,50,200)", width=2),
                fill="tozeroy",
                fillcolor="rgba(50,50,200,0.08)",
            ),
            go.Scatter(
                x=[char_d0, char_d0] if char_d0 > 0 else [None],
                y=[0, char_temp] if char_d0 > 0 else [None],
                mode="lines",
                name="炭化前線",
                line=dict(color="orange", width=2, dash="dash"),
            ),
        ],
        frames=frames,
    )

    # ── 炭化温度の水平線 ──
    fig.add_hline(
        y=char_temp,
        line_dash="dot",
        line_color="red",
        annotation_text=f"炭化温度 {char_temp:.0f}°C",
        annotation_position="top right",
    )

    # ── スライダー設定 ──
    sliders = [
        dict(
            active=0,
            currentvalue=dict(
                prefix="時刻: ",
                suffix=" 分",
                visible=True,
            ),
            pad=dict(b=10, t=50),
            steps=[
                dict(
                    args=[
                        [f.name],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                        ),
                    ],
                    label=f.name,
                    method="animate",
                )
                for f in frames
            ],
        )
    ]

    # ── 再生ボタン ──
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            x=0.0,
            xanchor="left",
            y=-0.22,
            yanchor="top",
            buttons=[
                dict(
                    label="▶ 再生",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=150, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="⏸ 停止",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                        ),
                    ],
                ),
            ],
        )
    ]

    fig.update_layout(
        title="厚み方向 温度プロファイル",
        xaxis_title="加熱面からの深さ [mm]",
        yaxis_title="温度 [°C]",
        yaxis=dict(range=[0, t_max_color]),
        xaxis=dict(range=[0, float(x_mm.max()) * 1.02]),
        sliders=sliders,
        updatemenus=updatemenus,
        height=500,
        margin=dict(b=120),
        legend=dict(x=0.7, y=0.98),
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# 炭化深さ 時刻歴チャート
# ---------------------------------------------------------------------------

def make_charring_chart(
    result: dict[str, Any],
    beta_ref: float = 0.65,
    tolerance: float = 0.15,
) -> Any:
    """炭化深さの時刻歴と Eurocode 5 参照線を Plotly で描画する。

    Parameters
    ----------
    result : dict
        solve() の返り値。
    beta_ref : float
        Eurocode 5 標準炭化速度 [mm/min]。
    tolerance : float
        許容誤差（±）の割合。デフォルト 0.15（±15%）。

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    times_min = result["times"] / 60.0
    char_mm = result["char_depths"] * 1000.0

    t_ref = np.linspace(0, float(times_min.max()), 200)
    beta_line = beta_ref * t_ref
    beta_upper = (1.0 + tolerance) * beta_ref * t_ref
    beta_lower = (1.0 - tolerance) * beta_ref * t_ref

    fig = go.Figure()

    # 許容帯（塗りつぶし）
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_ref, t_ref[::-1]]).tolist(),
        y=np.concatenate([beta_upper, beta_lower[::-1]]).tolist(),
        fill="toself",
        fillcolor="rgba(255,0,0,0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"許容範囲 ±{tolerance*100:.0f}%",
        hoverinfo="skip",
    ))

    # Eurocode 5 参照線
    fig.add_trace(go.Scatter(
        x=t_ref.tolist(),
        y=beta_line.tolist(),
        mode="lines",
        line=dict(color="red", dash="dash", width=1.5),
        name=f"Eurocode β₀={beta_ref} mm/min",
    ))

    # 解析結果
    fig.add_trace(go.Scatter(
        x=times_min.tolist(),
        y=char_mm.tolist(),
        mode="lines",
        line=dict(color="royalblue", width=2.5),
        name="解析結果（300°C 等温面）",
        hovertemplate="時刻: %{x:.1f}分<br>炭化深さ: %{y:.2f}mm<extra></extra>",
    ))

    fig.update_layout(
        title="炭化深さの時刻歴",
        xaxis_title="時間 [分]",
        yaxis_title="炭化深さ [mm]",
        xaxis=dict(range=[0, float(times_min.max())]),
        yaxis=dict(range=[0, None]),
        legend=dict(x=0.02, y=0.98),
        height=420,
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# 表面温度 時刻歴チャート
# ---------------------------------------------------------------------------

def make_surface_temp_chart(
    result: dict[str, Any],
    T_init: float = 20.0,
    insulation_limit: float = 100.0,
) -> Any:
    """加熱面・構造層表面・非加熱面温度と ISO 834 ガス温度の時刻歴を Plotly で描画する。

    Parameters
    ----------
    result : dict
        solve() の返り値。config キーがあれば構造層表面温度も描画する。
    T_init : float
        初期温度 [°C]。
    insulation_limit : float
        基準温度 [°C]。デフォルト 100°C。

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from clt_fire_sim.boundary import iso834_temperature

    times_min = result["times"] / 60.0
    T_mat = result["temperatures"]
    x_centers = result.get("x_centers")
    mesh = result.get("mesh")

    # 3D か 1D かを判定（mesh に ny 属性があれば 3D）
    is_3d = mesh is not None and hasattr(mesh, "ny")

    if is_3d:
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        T_heated   = np.array([T_mat[ti].reshape(nx, ny, nz)[0,  :, :].mean() for ti in range(len(T_mat))])
        T_unheated = np.array([T_mat[ti].reshape(nx, ny, nz)[-1, :, :].mean() for ti in range(len(T_mat))])
    else:
        T_heated   = T_mat[:, 0]
        T_unheated = T_mat[:, -1]

    # ── 構造層表面温度（config と x_centers があれば抽出）──────────────
    T_struct: Any = None
    struct_label = "構造層表面温度"
    config = result.get("config")
    if config is not None and x_centers is not None:
        try:
            spec = config.specimen
            eval_cfg = config.evaluation
            layer_idx = eval_cfg.structural_layer_index
            n_layers = len(spec.layers)
            if layer_idx < 0:
                layer_idx = n_layers + layer_idx
            layer_idx = int(np.clip(layer_idx, 0, n_layers - 1))
            layer_starts_m = np.cumsum(
                [0.0] + [la.thickness_mm / 1000.0 for la in spec.layers]
            )
            struct_x = layer_starts_m[layer_idx]
            sci = int(np.argmin(np.abs(x_centers - struct_x)))
            if is_3d:
                T_struct = np.array([
                    T_mat[ti].reshape(nx, ny, nz)[sci, :, :].mean()
                    for ti in range(len(T_mat))
                ])
            else:
                T_struct = T_mat[:, sci]
            struct_label = f"構造層表面温度（第{layer_idx + 1}層 火側面）"
        except Exception:
            T_struct = None

    # ISO 834 ガス温度
    T_iso = np.array([iso834_temperature(t) for t in times_min])

    fig = go.Figure()

    fig.add_hline(
        y=insulation_limit,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"{insulation_limit:.0f}°C",
        annotation_position="bottom right",
    )

    # ISO 834 ガス温度
    fig.add_trace(go.Scatter(
        x=times_min.tolist(),
        y=T_iso.tolist(),
        mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        name="ISO 834 ガス温度",
        hovertemplate="時刻: %{x:.1f}分<br>ISO 834: %{y:.0f}°C<extra></extra>",
    ))

    # ① 加熱面温度
    fig.add_trace(go.Scatter(
        x=times_min.tolist(),
        y=T_heated.tolist(),
        mode="lines",
        line=dict(color="firebrick", width=2.5),
        name="① 加熱面温度",
        hovertemplate="時刻: %{x:.1f}分<br>加熱面: %{y:.0f}°C<extra></extra>",
    ))

    # ② 構造層表面温度
    if T_struct is not None:
        fig.add_trace(go.Scatter(
            x=times_min.tolist(),
            y=np.asarray(T_struct).tolist(),
            mode="lines",
            line=dict(color="darkorange", width=2.5, dash="dot"),
            name=f"② {struct_label}",
            hovertemplate=f"時刻: %{{x:.1f}}分<br>{struct_label}: %{{y:.1f}}°C<extra></extra>",
        ))

    # ③ 非加熱面温度
    fig.add_trace(go.Scatter(
        x=times_min.tolist(),
        y=T_unheated.tolist(),
        mode="lines",
        line=dict(color="steelblue", width=2.5),
        name="③ 非加熱面温度",
        hovertemplate="時刻: %{x:.1f}分<br>非加熱面: %{y:.1f}°C<extra></extra>",
    ))

    fig.update_layout(
        title="温度時刻歴（加熱面 / 構造層表面 / 非加熱面）",
        xaxis_title="時間 [分]",
        yaxis_title="温度 [°C]",
        xaxis=dict(range=[0, float(times_min.max())]),
        yaxis=dict(range=[0, None]),
        legend=dict(x=0.02, y=0.98),
        height=420,
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# 温度場ヒートマップ（時間 × 深さ）
# ---------------------------------------------------------------------------

def make_temp_heatmap(
    result: dict[str, Any],
    max_frames: int = 200,
) -> Any:
    """時間×深さの温度場をヒートマップで表示する。

    x 軸: 時間 [分]、y 軸: 加熱面からの深さ [mm]、色: 温度 [°C]。
    大量の記録時刻がある場合はダウンサンプリングする。

    Parameters
    ----------
    result : dict
        solve() の返り値。
    max_frames : int
        ヒートマップの最大列数（時刻方向のダウンサンプリング上限）。

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    times_s = result["times"]
    times_min = times_s / 60.0
    x_mm = result["x_centers"] * 1000.0
    T_mat = result["temperatures"]

    # ダウンサンプリング（時刻方向）
    n_total = len(times_s)
    if n_total > max_frames:
        step = n_total // max_frames
        idx = np.arange(0, n_total, step)
        times_plot = times_min[idx]
        T_plot = T_mat[idx]
    else:
        times_plot = times_min
        T_plot = T_mat

    # T_plot が 3D の場合は最初の y,z スライスのみを使う
    if T_plot.ndim == 3:
        # shape: (Nt, Nx, Ny*Nz) → take first y,z
        T_plot = T_plot[:, :, 0]
    elif T_plot.ndim == 2 and T_plot.shape[1] != len(x_mm):
        # flat 3D: shape (Nt, Nx*Ny*Nz) → take every ny*nz step
        ny_nz = T_plot.shape[1] // len(x_mm)
        if ny_nz > 1:
            T_plot = T_plot[:, ::ny_nz]

    # Heatmap: x=時間, y=深さ（y軸反転で加熱面を上に）
    fig = go.Figure(data=go.Heatmap(
        z=T_plot.T.tolist(),  # shape: (Nx, Nt)
        x=times_plot.tolist(),
        y=x_mm.tolist(),
        colorscale="Inferno",
        colorbar=dict(title="温度 [°C]"),
        hovertemplate=(
            "時刻: %{x:.1f}分<br>"
            "深さ: %{y:.1f}mm<br>"
            "温度: %{z:.0f}°C<extra></extra>"
        ),
        zmin=0,
        zmax=max(float(T_plot.max()), 400.0),
    ))

    # 300°C 等温面（炭化前線）のオーバーレイ
    char_depths_mm = result["char_depths"] * 1000.0
    if n_total > max_frames:
        char_plot = char_depths_mm[idx]
    else:
        char_plot = char_depths_mm

    fig.add_trace(go.Scatter(
        x=times_plot.tolist(),
        y=char_plot.tolist(),
        mode="lines",
        line=dict(color="white", width=2, dash="dash"),
        name="炭化前線（300°C）",
        hovertemplate="時刻: %{x:.1f}分<br>炭化深さ: %{y:.2f}mm<extra></extra>",
    ))

    fig.update_layout(
        title="温度場ヒートマップ（時間 × 深さ）",
        xaxis_title="時間 [分]",
        yaxis_title="加熱面からの深さ [mm]",
        yaxis=dict(
            range=[float(x_mm.max()), 0],  # 加熱面（y=0）を上に
            autorange=False,
        ),
        height=450,
        legend=dict(x=0.02, y=0.02),
    )

    return fig


# ---------------------------------------------------------------------------
# 試験体断面図（層構成 + 温度計測点）
# ---------------------------------------------------------------------------

def make_specimen_diagram(config: Any) -> Any:
    """試験体の層構成を横断面図で描画し、グラフの温度計測点を重ねて表示する。

    Parameters
    ----------
    config : CLTConfig
        シミュレーション設定オブジェクト。

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from clt_fire_sim.materials import MATERIAL_DB

    spec = config.specimen
    layers = spec.layers
    n_layers = len(layers)

    # 構造層インデックス（-1 → 最終層）
    struct_idx = config.evaluation.structural_layer_index
    if struct_idx < 0:
        struct_idx = n_layers + struct_idx
    struct_idx = int(np.clip(struct_idx, 0, n_layers - 1))

    # 累積 x 座標（加熱面=0）
    thicknesses = [la.thickness_mm for la in layers]
    x_edges = [0.0] + list(np.cumsum(thicknesses))
    total_mm = x_edges[-1]
    pad = total_mm * 0.07  # 左右余白

    # 材料→塗りつぶし色（キーワード順一致）
    _MAT_COLORS: list[tuple[str, str]] = [
        ("air",      "rgba(190,225,245,0.85)"),
        ("noncomb",  "rgba(160,200,175,0.85)"),
        ("gypsum",   "rgba(230,215,215,0.85)"),
        ("steel",    "rgba(180,180,195,0.85)"),
        ("sugi",     "rgba(222,196,158,0.85)"),
        ("hinoki",   "rgba(215,188,145,0.85)"),
        ("oak",      "rgba(195,165,120,0.85)"),
        ("pine",     "rgba(218,192,150,0.85)"),
        ("wood",     "rgba(210,182,140,0.85)"),
        ("clt",      "rgba(200,174,130,0.85)"),
    ]

    def _color(mat: str) -> str:
        ml = mat.lower()
        for key, c in _MAT_COLORS:
            if key in ml:
                return c
        return "rgba(200,200,200,0.85)"

    fig = go.Figure()

    # ── 各層の矩形（y: 0 → 1）──────────────────────────────────────────
    for i, layer in enumerate(layers):
        x0, x1 = x_edges[i], x_edges[i + 1]
        is_struct = i == struct_idx
        fig.add_shape(
            type="rect", x0=x0, y0=0, x1=x1, y1=1,
            fillcolor=_color(layer.material),
            line=dict(color="darkorange" if is_struct else "#aaa",
                      width=3.0 if is_struct else 1.0),
        )
        mat_disp = MATERIAL_DB.get(layer.material, {}).get("name", layer.material)
        w = x1 - x0
        # フォントサイズを幅比率で調整（最小 8pt）
        fs = max(8, min(12, int(w / total_mm * n_layers * 13)))
        x_mid = (x0 + x1) / 2
        fig.add_annotation(
            x=x_mid, y=0.5,
            text=f"第{i+1}層<br>{mat_disp}<br>{w:.0f}mm",
            showarrow=False, font=dict(size=fs), align="center",
        )

    # ── ① 加熱面温度（x=0）──────────────────────────────────────────────
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=1.38,
                  line=dict(color="firebrick", width=2.5))
    fig.add_annotation(
        x=0, y=1.45, xanchor="center",
        text="<b>① 加熱面温度</b>",
        showarrow=False, font=dict(color="firebrick", size=12),
    )

    # ── ② 構造層表面温度（構造層の火側面）────────────────────────────────
    x_struct = x_edges[struct_idx]
    # ① と重なる場合（struct_idx=0）は下側に注釈を出す
    if struct_idx == 0:
        fig.add_shape(type="line", x0=x_struct, y0=-0.38, x1=x_struct, y1=1,
                      line=dict(color="darkorange", width=2.5, dash="dot"))
        fig.add_annotation(
            x=x_struct, y=-0.45, xanchor="left",
            text=f"<b>② 構造層表面温度（第{struct_idx+1}層 火側面）</b>",
            showarrow=False, font=dict(color="darkorange", size=11),
        )
    else:
        fig.add_shape(type="line", x0=x_struct, y0=-0.38, x1=x_struct, y1=1,
                      line=dict(color="darkorange", width=2.5, dash="dot"))
        fig.add_annotation(
            x=x_struct, y=-0.45, xanchor="center",
            text=f"<b>② 構造層表面温度（第{struct_idx+1}層 火側面）</b>",
            showarrow=False, font=dict(color="darkorange", size=11),
        )

    # ── ③ 非加熱面温度（x=total_mm）─────────────────────────────────────
    fig.add_shape(type="line", x0=total_mm, y0=0, x1=total_mm, y1=1.38,
                  line=dict(color="steelblue", width=2.5))
    fig.add_annotation(
        x=total_mm, y=1.45, xanchor="center",
        text="<b>③ 非加熱面温度</b>",
        showarrow=False, font=dict(color="steelblue", size=12),
    )

    # 🔥 / ❄️
    fig.add_annotation(x=-pad * 0.6, y=0.5, text="🔥", showarrow=False,
                       font=dict(size=20), xanchor="center")
    fig.add_annotation(x=total_mm + pad * 0.6, y=0.5, text="❄️", showarrow=False,
                       font=dict(size=20), xanchor="center")

    # 「構造層」吹き出し（矩形上部）
    fig.add_annotation(
        x=(x_edges[struct_idx] + x_edges[struct_idx + 1]) / 2,
        y=1.03, xanchor="center", yanchor="bottom",
        text=f"◀ 構造層（第{struct_idx+1}層）▶",
        showarrow=False, font=dict(color="darkorange", size=11),
    )

    fig.update_layout(
        title=dict(text=f"試験体断面図：{spec.name}　総厚 {total_mm:.0f} mm", font=dict(size=14)),
        xaxis=dict(
            title="加熱面からの距離 [mm]",
            range=[-pad, total_mm + pad],
            showgrid=True, gridcolor="#eee", zeroline=False,
        ),
        yaxis=dict(
            range=[-0.75, 1.75],
            showticklabels=False, showgrid=False, zeroline=False, fixedrange=True,
        ),
        height=340,
        margin=dict(l=10, r=10, t=55, b=20),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig
