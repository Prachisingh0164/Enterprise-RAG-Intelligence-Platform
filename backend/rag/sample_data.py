"""SYNTHETIC sample enterprise documents + a labelled evaluation set derived from them.
Every document says "synthetic sample document". Replace with real public filings (SEC EDGAR, arXiv) for real use."""
import json
from pathlib import Path

# file -> (allowed_roles, eval user role, pages); fact = (sentence, question, expected keyword)
DOCS = {
 "employee_handbook.pdf": (["all"], "employee", "Employee Handbook", [
  [("Full-time employees receive 24 days of paid annual leave each calendar year.", "How many days of paid annual leave do full-time employees get?", "24 days"),
   ("Unused annual leave of up to 5 days can be carried over to the next year.", "How many unused leave days can be carried over?", "5 days"),
   ("Core working hours are 10:00 to 16:00 Monday to Friday.", "What are the core working hours?", "10:00"),
   ("Sick leave requires a medical certificate after 3 consecutive days.", "When is a medical certificate required for sick leave?", "3 consecutive days")],
  [("Employees may work remotely up to 3 days per week with manager approval.", "How many days per week can employees work remotely?", "3 days"),
   ("Business travel expenses must be submitted within 30 days of the trip.", "Within how many days must travel expenses be submitted?", "30 days"),
   ("Meals during travel are reimbursed up to 45 USD per day.", "What is the daily meal reimbursement limit for travel?", "45"),
   ("The company offers a learning budget of 1200 USD per employee per year.", "How much is the annual learning budget per employee?", "1200")]]),
 "it_security_policy.docx": (["all"], "employee", "IT Security Policy", [
  [("Passwords must be at least 14 characters long and changed every 90 days.", "What is the minimum password length?", "14 characters"),
   ("Multi-factor authentication is mandatory for all VPN and email access.", "Where is multi-factor authentication mandatory?", "vpn"),
   ("Lost or stolen devices must be reported to the IT helpdesk within 2 hours.", "Within what time must a lost device be reported to the helpdesk?", "2 hours"),
   ("Company laptops must use full-disk encryption enabled by IT.", "What encryption must company laptops use?", "full-disk")],
  [("Phishing emails should be reported using the Report Phish button in the mail client.", "How should phishing emails be reported?", "report phish"),
   ("Security incidents are classified as P1, P2 or P3, and P1 incidents require a response within 15 minutes.", "What is the response time for P1 security incidents?", "15 minutes"),
   ("Personal USB storage devices are not permitted on company machines.", "Are personal USB storage devices allowed on company machines?", "not permitted"),
   ("Software installation requests are approved by the IT security team within 3 business days.", "How long does approval of software installation requests take?", "3 business days")]]),
 "hr_compensation_confidential.pdf": (["hr"], "hr", "HR Compensation Guide", [
  [("Salary band L3 ranges from 52000 to 68000 USD per year.", "What is the salary range of band L3?", "52000"),
   ("Salary band L4 ranges from 68000 to 92000 USD per year.", "What is the salary range of band L4?", "92000"),
   ("The annual performance bonus target is 10 percent of base salary for levels L3 and L4.", "What is the annual bonus target for L3 and L4?", "10 percent")],
  [("Performance reviews take place twice a year, in June and December.", "When do performance reviews take place?", "june and december"),
   ("Employees rated Exceeds Expectations are eligible for a 6 percent salary increase.", "What salary increase do Exceeds Expectations employees get?", "6 percent"),
   ("Stock option grants vest over 4 years with a 1 year cliff.", "Over how many years do stock options vest?", "4 years")]]),
 "finance_q3_report.pdf": (["finance"], "finance", "Finance Q3 Report", [
  [("Q3 revenue was 48.2 million USD, up 12 percent year over year.", "What was Q3 revenue?", "48.2"),
   ("Gross margin in Q3 was 61 percent.", "What was the gross margin in Q3?", "61 percent"),
   ("Operating expenses totalled 21.4 million USD in Q3.", "What were the operating expenses in Q3?", "21.4")],
  [("The company ended Q3 with 412 employees across 6 offices.", "How many employees did the company have at the end of Q3?", "412"),
   ("Cash and equivalents stood at 95.6 million USD at the end of the quarter.", "How much cash did the company have at the end of the quarter?", "95.6"),
   ("Full-year revenue guidance was raised to 190 million USD.", "What is the full-year revenue guidance?", "190")]]),
 "product_manual.txt": (["all"], "employee", "X200 Product Manual", [
  [("To pair the X200 headset, hold the power button for 5 seconds until the light blinks blue.", "How do I pair the X200 headset?", "5 seconds"),
   ("The X200 battery lasts up to 30 hours on a single charge.", "How long does the X200 battery last?", "30 hours"),
   ("A full charge of the X200 takes about 2 hours using the USB-C cable.", "How long does it take to fully charge the X200?", "2 hours")],
  [("To factory reset the X200, press and hold volume up and power for 10 seconds.", "How do I factory reset the X200?", "10 seconds"),
   ("The X200 carries a 24 month limited warranty.", "How long is the X200 warranty?", "24 month")]]),
}
OUT_OF_SCOPE = ["What is the capital of France?", "Who won the football world cup in 2018?", "What is the CEO’s favourite colour?",
                "How do I bake sourdough bread?", "What is the price of Bitcoin today?"]


def _paras(title, page):
    return [f"{title}, synthetic sample document."] + [" ".join(f[0] for f in page[i:i + 2]) for i in range(0, len(page), 2)]


def write_files(out: Path, eval_file: Path):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    import docx
    out.mkdir(exist_ok=True); eval_file.parent.mkdir(exist_ok=True); ev = []
    for name, (roles, urole, title, pages) in DOCS.items():
        ext = name.rsplit(".", 1)[1]
        if ext == "pdf":
            c = canvas.Canvas(str(out / name), pagesize=A4)
            for page in pages:
                y = 800; c.setFont("Helvetica", 11)
                for p in _paras(title, page):
                    for line in simpleSplit(p, "Helvetica", 11, 495): c.drawString(50, y, line); y -= 15
                    y -= 8
                c.showPage()
            c.save()
        elif ext == "docx":
            d = docx.Document()
            for i, page in enumerate(pages):
                if i: d.add_page_break()
                for p in _paras(title, page): d.add_paragraph(p)
            d.save(str(out / name))
        else:
            (out / name).write_text("\f".join("\n".join(_paras(title, pg)) for pg in pages), encoding="utf-8")
        for pno, page in enumerate(pages, 1):
            for _, q, kw in page: ev.append(dict(question=q, doc=name, page=pno, keyword=kw, user_role=urole, restricted=roles != ["all"]))
    eval_file.write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    return {n: v[0] for n, v in DOCS.items()}
