#!/usr/bin/env python3
import markdown, re
src = open('/home/tiger/xiaoxuan/FoldAgent/report/REPORT.md').read()
# inline the figures after their relevant sections
src = src.replace('## 3. Behavioral signatures', '![](assets/fig1_scores.png)\n\n## 3. Behavioral signatures')
src = src.replace('## 4. Methodological finding', '![](assets/fig2_behavior.png)\n\n## 4. Methodological finding')
body = markdown.markdown(src, extensions=['tables'])
html = f"""<html><head><meta charset='utf-8'><style>
body {{ font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.5pt; margin: 2.2cm; line-height: 1.45; color: #1a1a1a; }}
h1 {{ font-size: 16pt; }} h2 {{ font-size: 12.5pt; margin-top: 1.2em; border-bottom: 1px solid #ccc; }}
table {{ border-collapse: collapse; margin: 0.8em 0; font-size: 9.5pt; }}
th, td {{ border: 1px solid #999; padding: 3px 8px; }} th {{ background: #f0f0f0; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f5f5f5; }}
img {{ max-width: 100%; }}
</style></head><body>{body}</body></html>"""
open('/home/tiger/xiaoxuan/FoldAgent/report/REPORT.html', 'w').write(html)
from weasyprint import HTML
HTML('/home/tiger/xiaoxuan/FoldAgent/report/REPORT.html', base_url='/home/tiger/xiaoxuan/FoldAgent/report/').write_pdf('/home/tiger/xiaoxuan/FoldAgent/report/REPORT.pdf')
print('PDF written')
