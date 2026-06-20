import os
import re

def test_html_structure():
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    assert os.path.exists(html_path), "index.html does not exist in frontend folder"
    
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check that stylesheets and script tags are correctly referenced
    assert 'href="index.css"' in content, "index.css is not linked in index.html"
    assert 'src="app.js"' in content, "app.js is not referenced in index.html"
    
    # Check for Meta Design components mentioned in index.html
    assert 'class="promo-banner"' in content, "Promo banner element is missing"
    assert 'class="top-nav"' in content, "Sticky navigation bar is missing"
    assert 'class="card-checkout-summary"' in content, "Config summaries checking block is missing"
    assert 'class="why-buy-tile"' in content, "Excision core reassurance grid tiles are missing"
    assert 'class="faq-accordion-item"' in content, "FAQ accordion widgets are missing"

def test_css_design_system_variables():
    css_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.css")
    assert os.path.exists(css_path), "index.css does not exist in frontend folder"
    
    with open(css_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check for important color definitions from DESIGN.md
    assert re.search(r"--primary:\s*#0064e0", content), "Primary Cobalt color token #0064e0 is missing or mismatched"
    assert re.search(r"--primary-deep:\s*#0457cb", content), "Primary Deep Cobalt color token #0457cb is missing or mismatched"
    assert re.search(r"--fb-blue:\s*#1876f2", content), "Facebook Blue color token #1876f2 is missing or mismatched"
    assert re.search(r"--canvas:\s*#ffffff", content), "Canvas White color token #ffffff is missing or mismatched"
    assert re.search(r"--ink-deep:\s*#0a1317", content), "Ink Deep color token #0a1317 is missing or mismatched"
    
    # Check for rounded corner scales from DESIGN.md
    assert re.search(r"--rounded-lg:\s*8px", content), "Rounded LG token 8px is missing or mismatched"
    assert re.search(r"--rounded-xl:\s*16px", content), "Rounded XL token 16px is missing or mismatched"
    assert re.search(r"--rounded-xxxl:\s*32px", content), "Rounded XXXL token 32px is missing or mismatched"
    assert re.search(r"--rounded-full:\s*100px", content), "Rounded Full token 100px is missing or mismatched"
    
    # Check for typography parameters
    assert "Optimistic VF" in content, "Optimistic VF font family token is missing"
    assert "Montserrat" in content, "Montserrat fallback font is missing"
