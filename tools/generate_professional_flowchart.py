from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "so_do_quy_trinh_tu_gui_do_professional.svg"

W, H = 2400, 1550
FONT = "'Segoe UI', Arial, sans-serif"


def tspan_lines(text, x, y, size=24, color="#0f172a", weight=500, line_gap=1.18):
    lines = text.split("\n")
    total = (len(lines) - 1) * size * line_gap
    first_y = y - total / 2
    parts = [
        f'<text x="{x}" y="{first_y:.1f}" text-anchor="middle" '
        f'dominant-baseline="middle" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">'
    ]
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * line_gap
        parts.append(f'<tspan x="{x}" dy="{dy:.1f}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "\n".join(parts)


def rounded_rect(x, y, w, h, text, fill="#ffffff", stroke="#475569", sw=3,
                 rx=18, size=24, color="#0f172a", weight=500, extra_class=""):
    return f"""
<g class="{extra_class}">
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>
  {tspan_lines(text, x + w / 2, y + h / 2, size=size, color=color, weight=weight)}
</g>"""


def pill(cx, cy, w, h, text, fill="#ffffff", stroke="#334155", sw=3,
         size=25, color="#0f172a", weight=600):
    return f"""
<g>
  <rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" rx="{h / 2}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>
  {tspan_lines(text, cx, cy, size=size, color=color, weight=weight)}
</g>"""


def diamond(cx, cy, w, h, text, fill="#fff7ed", stroke="#d97706", sw=3,
            size=22, color="#0f172a", weight=600):
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    return f"""
<g>
  <polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />
  {tspan_lines(text, cx, cy, size=size, color=color, weight=weight)}
</g>"""


def lane(x, y, w, h, title, fill, stroke, subtitle=None):
    header_h = 64
    sub = f'<text x="{x + 26}" y="{y + 47}" font-family="{FONT}" font-size="20" fill="#475569">{escape(subtitle)}</text>' if subtitle else ""
    return f"""
<g>
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
  <rect x="{x}" y="{y}" width="{w}" height="{header_h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <path d="M {x} {y + header_h - 22} Q {x} {y + header_h} {x + 22} {y + header_h} L {x + w - 22} {y + header_h} Q {x + w} {y + header_h} {x + w} {y + header_h - 22}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <text x="{x + 26}" y="{y + 31}" font-family="{FONT}" font-size="26" font-weight="700" fill="#0f172a">{escape(title)}</text>
  {sub}
</g>"""


def path(points, label=None, label_pos=None, color="#334155", sw=3,
         dashed=False, marker=True):
    d = "M " + " L ".join(f"{x} {y}" for x, y in points)
    dash = ' stroke-dasharray="9 8"' if dashed else ""
    mk = ' marker-end="url(#arrow)"' if marker else ""
    label_svg = ""
    if label:
        lx, ly = label_pos if label_pos else points[len(points) // 2]
        label_svg = (
            f'<g><rect x="{lx - 42}" y="{ly - 16}" width="84" height="28" rx="14" '
            f'fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{lx}" y="{ly + 1}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="{FONT}" font-size="18" font-weight="600" fill="{color}">{escape(label)}</text></g>'
        )
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{dash}{mk}/>{label_svg}'


def dot(cx, cy, text, fill="#ffffff", stroke="#64748b"):
    return f"""
<g>
  <circle cx="{cx}" cy="{cy}" r="22" fill="{fill}" stroke="{stroke}" stroke-width="3"/>
  <text x="{cx}" y="{cy + 1}" text-anchor="middle" dominant-baseline="middle" font-family="{FONT}" font-size="19" font-weight="700" fill="#334155">{escape(text)}</text>
</g>"""


def main():
    x1, x2, x3 = 80, 830, 1580
    lane_w, lane_y, lane_h = 710, 360, 980
    c1, c2, c3 = x1 + lane_w / 2, x2 + lane_w / 2, x3 + lane_w / 2

    svg = [f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">
    <path d="M 2 2 L 10 6 L 2 10 z" fill="#334155"/>
  </marker>
  <filter id="shadow" x="-8%" y="-8%" width="116%" height="116%">
    <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.13"/>
  </filter>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{W/2}" y="58" text-anchor="middle" font-family="{FONT}" font-size="38" font-weight="800" fill="#0f172a">Quy trình gửi và lấy đồ bằng nhận diện khuôn mặt + vân tay</text>
<text x="{W/2}" y="96" text-anchor="middle" font-family="{FONT}" font-size="22" font-weight="500" fill="#475569">Bản vẽ trình bày lại để nhập/chỉnh sửa trong EdrawMax</text>
"""]

    # Top common flow
    svg.append(pill(W / 2, 142, 190, 64, "Start"))
    svg.append(rounded_rect(W / 2 - 260, 190, 520, 72, "Hiển thị trạng thái\n(Số tủ trống / đang sử dụng)",
                            fill="#eff6ff", stroke="#2563eb", size=24, weight=650))
    svg.append(diamond(W / 2, 330, 350, 104, "Chọn thao tác", fill="#fefce8", stroke="#ca8a04", size=25))
    svg.append(path([(W / 2, 174), (W / 2, 190)]))
    svg.append(path([(W / 2, 262), (W / 2, 278)]))
    svg.append(path([(W / 2, 382), (W / 2, 384)], marker=False))

    # Lanes
    svg.append(lane(x1, lane_y, lane_w, lane_h, "Gửi đồ / Đăng ký", "#dff6f3", "#0f766e",
                    "Tạo hồ sơ khuôn mặt, vân tay và gán locker"))
    svg.append(lane(x2, lane_y, lane_w, lane_h, "Lấy tạm thời", "#e8f7e8", "#16a34a",
                    "Mở tủ nhưng giữ trạng thái đang sử dụng"))
    svg.append(lane(x3, lane_y, lane_w, lane_h, "Lấy hoàn toàn", "#fff1e6", "#ea580c",
                    "Mở tủ, xóa dữ liệu và trả locker về trống"))

    # Arrows from action choice
    svg.append(path([(1025, 330), (435, 330), (435, 424)], label="Gửi đồ", label_pos=(720, 330)))
    svg.append(path([(1200, 382), (1200, 424)], label="Tạm thời", label_pos=(1268, 400)))
    svg.append(path([(1375, 330), (1935, 330), (1935, 424)], label="Hoàn toàn", label_pos=(1605, 330)))

    # Lane 1: registration
    svg.append(rounded_rect(c1 - 170, 424, 340, 70, "Kiểm tra locker trống", stroke="#0f766e"))
    svg.append(diamond(c1, 548, 310, 112, "Còn locker\ntrống?", fill="#ecfeff", stroke="#0891b2"))
    svg.append(rounded_rect(c1 - 170, 636, 340, 70, "Mở camera đăng ký", stroke="#0f766e"))
    svg.append(rounded_rect(c1 - 190, 742, 380, 76, "Detect + kiểm tra\nchất lượng ảnh", stroke="#0f766e", size=23))
    svg.append(diamond(c1, 870, 330, 118, "Ảnh/căn mặt\nhợp lệ?", fill="#ecfeff", stroke="#0891b2", size=21))
    svg.append(rounded_rect(c1 - 178, 958, 356, 72, "Quét vân tay đăng ký", stroke="#0f766e"))
    svg.append(diamond(c1, 1090, 320, 112, "Vân tay\nhợp lệ?", fill="#ecfeff", stroke="#0891b2", size=21))
    svg.append(rounded_rect(c1 - 190, 1178, 380, 78, "Huấn luyện user\n(cập nhật dataset + embedding)",
                            stroke="#0f766e", size=21))
    svg.append(rounded_rect(c1 - 170, 1278, 340, 66, "Đăng ký thành công", fill="#ecfdf5", stroke="#059669", size=23, weight=650))
    svg.append(rounded_rect(x1 + 36, 646, 245, 78, "Thông báo:\nhết chỗ trống", fill="#fef2f2", stroke="#dc2626", size=21, weight=650))
    svg.append(rounded_rect(x1 + 36, 780, 245, 70, "Đăng ký thất bại", fill="#fef2f2", stroke="#dc2626", size=22, weight=650))

    svg.append(path([(c1, 494), (c1, 492)]))
    svg.append(path([(c1, 604), (c1, 636)], label="Có", label_pos=(475, 625)))
    svg.append(path([(280, 548), (202, 548), (202, 646)], label="Không", label_pos=(252, 548)))
    svg.append(path([(202, 724), (202, 780)]))
    svg.append(path([(c1, 706), (c1, 742)]))
    svg.append(path([(c1, 818), (c1, 811)]))
    svg.append(path([(c1, 929), (c1, 958)], label="Có", label_pos=(475, 946)))
    svg.append(path([(274, 870), (202, 870), (202, 850)], label="Không", label_pos=(255, 870)))
    svg.append(path([(c1, 1030), (c1, 1034)]))
    svg.append(path([(c1, 1146), (c1, 1178)], label="Có", label_pos=(475, 1161)))
    svg.append(path([(275, 1090), (202, 1090), (202, 850)], label="Không", label_pos=(252, 1090)))
    svg.append(path([(c1, 1256), (c1, 1278)]))

    # Lane 2: temporary retrieval
    r2 = x2 + 532
    svg.append(rounded_rect(c2 - 176, 424, 352, 70, "Mở camera nhận diện", stroke="#16a34a"))
    svg.append(rounded_rect(c2 - 176, 528, 352, 70, "Detect + State", stroke="#16a34a"))
    svg.append(rounded_rect(c2 - 190, 632, 380, 74, "Embedding + so khớp mẫu", stroke="#16a34a", size=23))
    svg.append(diamond(c2, 760, 332, 118, "Đúng người gửi?\n(confidence đạt)", fill="#f0fdf4", stroke="#16a34a", size=20))
    svg.append(rounded_rect(r2 - 150, 836, 300, 70, "Nhận diện lại\n/ recheck", stroke="#16a34a", size=22))
    svg.append(diamond(r2, 980, 282, 108, "Recheck\nđạt?", fill="#f0fdf4", stroke="#16a34a", size=21))
    svg.append(rounded_rect(r2 - 150, 1058, 300, 68, "Quét vân tay\nxác minh", stroke="#16a34a", size=22))
    svg.append(diamond(r2, 1188, 282, 106, "Vân tay\nđúng?", fill="#f0fdf4", stroke="#16a34a", size=21))
    svg.append(rounded_rect(c2 - 310, 1246, 360, 78, "Mở tủ và giữ trạng thái\nđang sử dụng", fill="#ecfdf5", stroke="#059669", size=21, weight=650))
    svg.append(rounded_rect(r2 - 115, 1274, 230, 70, "Từ chối mở tủ", fill="#fef2f2", stroke="#dc2626", size=21, weight=650))

    svg.append(path([(c2, 494), (c2, 528)]))
    svg.append(path([(c2, 598), (c2, 632)]))
    svg.append(path([(c2, 706), (c2, 701)]))
    svg.append(path([(c2, 819), (c2, 1246)], label="Có", label_pos=(1226, 868)))
    svg.append(path([(1351, 760), (r2, 760), (r2, 836)], label="Không", label_pos=(1458, 732)))
    svg.append(path([(r2, 906), (r2, 926)]))
    svg.append(path([(r2 - 141, 980), (c2 + 40, 980), (c2 + 40, 1246)], label="Có", label_pos=(1296, 948)))
    svg.append(path([(r2, 1034), (r2, 1058)], label="Không", label_pos=(1428, 1048)))
    svg.append(path([(r2, 1126), (r2, 1135)]))
    svg.append(path([(r2 - 141, 1188), (c2 + 40, 1188), (c2 + 40, 1246)], label="Có", label_pos=(1296, 1156)))
    svg.append(path([(r2, 1241), (r2, 1274)], label="Không", label_pos=(1428, 1252)))

    # Lane 3: complete retrieval
    r3 = x3 + 532
    svg.append(rounded_rect(c3 - 176, 424, 352, 70, "Mở camera nhận diện", stroke="#ea580c"))
    svg.append(rounded_rect(c3 - 176, 528, 352, 70, "Detect + State", stroke="#ea580c"))
    svg.append(rounded_rect(c3 - 190, 632, 380, 74, "Embedding + so khớp mẫu", stroke="#ea580c", size=23))
    svg.append(diamond(c3, 760, 332, 118, "Đúng người gửi?\n(confidence đạt)", fill="#fff7ed", stroke="#ea580c", size=20))
    svg.append(rounded_rect(r3 - 150, 836, 300, 70, "Nhận diện lại\n/ recheck", stroke="#ea580c", size=22))
    svg.append(diamond(r3, 980, 282, 108, "Recheck\nđạt?", fill="#fff7ed", stroke="#ea580c", size=21))
    svg.append(rounded_rect(r3 - 150, 1058, 300, 68, "Quét vân tay\nxác minh", stroke="#ea580c", size=22))
    svg.append(diamond(r3, 1188, 282, 106, "Vân tay\nđúng?", fill="#fff7ed", stroke="#ea580c", size=21))
    svg.append(rounded_rect(c3 - 310, 1246, 360, 78, "Mở tủ và xóa dữ liệu\nngười gửi", fill="#ecfdf5", stroke="#059669", size=21, weight=650))
    svg.append(rounded_rect(r3 - 115, 1274, 230, 70, "Từ chối lấy đồ", fill="#fef2f2", stroke="#dc2626", size=21, weight=650))

    svg.append(path([(c3, 494), (c3, 528)]))
    svg.append(path([(c3, 598), (c3, 632)]))
    svg.append(path([(c3, 706), (c3, 701)]))
    svg.append(path([(c3, 819), (c3, 1246)], label="Có", label_pos=(1976, 868)))
    svg.append(path([(2101, 760), (r3, 760), (r3, 836)], label="Không", label_pos=(2208, 732)))
    svg.append(path([(r3, 906), (r3, 926)]))
    svg.append(path([(r3 - 141, 980), (c3 + 40, 980), (c3 + 40, 1246)], label="Có", label_pos=(2046, 948)))
    svg.append(path([(r3, 1034), (r3, 1058)], label="Không", label_pos=(2178, 1048)))
    svg.append(path([(r3, 1126), (r3, 1135)]))
    svg.append(path([(r3 - 141, 1188), (c3 + 40, 1188), (c3 + 40, 1246)], label="Có", label_pos=(2046, 1156)))
    svg.append(path([(r3, 1241), (r3, 1274)], label="Không", label_pos=(2178, 1252)))

    # Common close-out
    footer_y = 1422
    svg.append(rounded_rect(655, footer_y, 1090, 72, "Ghi log sự kiện, cập nhật giao diện và quay về màn hình chính",
                            fill="#f1f5f9", stroke="#64748b", size=24, weight=650))
    svg.append(dot(42, 240, "A", fill="#eff6ff", stroke="#2563eb"))
    svg.append(dot(42, footer_y + 36, "A", fill="#f1f5f9", stroke="#64748b"))
    svg.append(path([(42, footer_y + 14), (42, 290), (940, 290), (940, 243)], dashed=True, color="#64748b", sw=2))
    svg.append(path([(42, 262), (42, 270), (940, 270), (940, 243)], dashed=True, color="#64748b", sw=2, marker=False))

    # Terminals to footer
    terminals = [
        (202, 850), (c1, 1344), (c2, 1324), (r2, 1344), (c3, 1324), (r3, 1344)
    ]
    targets = [
        (780, footer_y), (910, footer_y), (1130, footer_y), (1255, footer_y), (1485, footer_y), (1630, footer_y)
    ]
    for (sx, sy), (tx, ty) in zip(terminals, targets):
        svg.append(path([(sx, sy), (sx, ty - 36), (tx, ty)], color="#64748b", sw=2))

    # Small legend
    legend_x, legend_y = 1825, 1460
    svg.append(f"""
<g>
  <rect x="{legend_x}" y="{legend_y - 36}" width="430" height="56" rx="18" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/>
  <rect x="{legend_x + 20}" y="{legend_y - 20}" width="28" height="18" rx="5" fill="#ffffff" stroke="#475569" stroke-width="2"/>
  <text x="{legend_x + 58}" y="{legend_y - 6}" font-family="{FONT}" font-size="18" fill="#334155">Xử lý</text>
  <polygon points="{legend_x + 135},{legend_y - 21} {legend_x + 155},{legend_y - 11} {legend_x + 135},{legend_y - 1} {legend_x + 115},{legend_y - 11}" fill="#fff7ed" stroke="#d97706" stroke-width="2"/>
  <text x="{legend_x + 168}" y="{legend_y - 6}" font-family="{FONT}" font-size="18" fill="#334155">Quyết định</text>
  <rect x="{legend_x + 280}" y="{legend_y - 20}" width="28" height="18" rx="5" fill="#ecfdf5" stroke="#059669" stroke-width="2"/>
  <text x="{legend_x + 318}" y="{legend_y - 6}" font-family="{FONT}" font-size="18" fill="#334155">Thành công</text>
</g>
""")

    svg.append("</svg>\n")
    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
