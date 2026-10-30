# Prompt Library for Video Script / Storyboard
## Create a Professional Promotional Explainer Video for NeuroMed AI – Complete Clinical Decision Platform for Wilson's Disease & Rare Metabolic Disorders

### TOOLS
- Playwright (Chromium) to drive and record the browser: viewport 1920×1080, deviceScaleFactor 1, 30 fps.
- edge-tts for narration.
- FFmpeg for editing, zooms, blur, captions, and audio mixing.
- Python or Node for orchestration. Keep all sources in a project folder so the video can be rebuilt.

---

### CREDENTIALS (SECURITY & MEDICAL PRIVACY — MANDATORY)
- Read login details from clinical test environment. Never hard-code, print, log, speak, caption, or store real doctor credentials, hospital tokens, or real patient identifiers anywhere.
- On screen, the clinician email field must display "dr.demo@hospital.org" and the password field must show only masked dots. Achieve this with a post-production overlay that covers real field values in every frame where they are visible.
- On screen the user profile name should appear as "Dr. Demo Clinician (Chief Medical Officer)".
- Patient names must appear as approved mock profiles (e.g., "Demo Patient #WD-101").
- Disable Playwright tracing, and delete any temporary recordings that show sensitive data once the final video is approved.

---

### VISUAL RULES
- Preserve the actual backgrounds, deep clinical blues, emerald indicators, hospital logos, headers, and layouts of the NeuroMed AI platform. Do not redesign any screen, and do not invent artificial app transitions. Use straight cuts between real pages; if an assessment opens in a new tab, cut to that tab.
- Use the real clinical features exactly as they appear on screen.
- Add a visible cursor (headless recording does not show one) that moves smoothly (ease-in-out, 400–700 ms) to each click target, with a subtle click ripple.
- Highlights: a soft rounded outline (3 px, semi-transparent) plus a light glow around the element about to be clicked, appearing 0.5 s before the click. Highlights must be overlays only; never change the page itself.
- Zooms: gentle, 1.0× to at most 1.5×, over 0.8–1.2 s, centred on the active diagnostic element, and return to 1.0× before cuts.
- Blur personal health information (PHI) on authenticated pages: patient names, phone numbers, addresses, and medical record numbers. Use a Gaussian blur strong enough to be unreadable. List each blurred region in a log so it can be reviewed.

---

### VOICE
- Primary voice: en-IN-NeerjaNeural (female, Indian English). If it is unavailable, use en-IN-PrabhatNeural. Check with: `edge-tts --list-voices | grep en-IN`
- Settings: `--rate=-8% --volume=+0% --pitch=+0Hz`, for a calm, authoritative, reassuring medical director delivery.
- Generate each line separately, trim leading and trailing silence, and place each line 0.4 s after its segment starts. If a line runs longer than its segment, slow the screen action down rather than speeding up the voice.
- Voice level: -16 LUFS integrated, true peak no higher than -1.5 dBTP.

---

### BACKGROUND MUSIC
- A soft, calm modern instrumental (gentle piano and light acoustic ambient, hopeful and respectful in tone). Use only a royalty-free or properly licensed track.
- Level: about -30 LUFS under speech. Use sidechain ducking so the music drops by about 8 dB while the narrator speaks and rises gently in pauses.
- Fade in over 2 s at the start and fade out over 3 s at the end.

---

### CAPTIONS
- Burn captions into the video and also export a matching .srt file.
- Font: clean sans-serif (Noto Sans, Inter, or Arial), about 44 px, white text on a 60%-opacity dark rounded box, bottom-centre, keeping 80 px clear of the bottom edge.
- Maximum 2 lines and about 42 characters per line. Each caption appears with its spoken line and disappears 0.2 s after it ends.
- Captions show the official clinical platform names ("NeuroMed AI").

---

### VIDEO OBJECTIVE
The video should attract new healthcare clients such as Hospital Networks, Chief Medical Officers, Heads of Hepatology and Neurology, Diagnostic Pathology Chains, and Rare Disease Centers who are looking for a simple, secure, and complete digital clinical platform to eliminate diagnostic delays, reduce misdiagnoses, and standardize disease management.

The tone should be:
● Professional  
● Trustworthy  
● Modern  
● Easy to understand  
● Suitable for executive healthcare marketing (LinkedIn, Healthcare Summits, Medical Conferences, Hospital Board Presentations)  

---

### TARGET AUDIENCE
Decision-makers of:
● Multi-Specialty Hospital Networks & Medical Centers  
● Chief Medical Officers (CMOs) & Clinical Directors  
● Hepatologists, Gastroenterologists & Neurologists  
● Diagnostic Pathology Networks & Commercial Clinical Labs  
● Rare Disease & Metabolic Health Foundations  

---

### VIDEO DURATION
Approximately 3 minutes (can also be adapted into a 1-minute executive cut).

---

### VIDEO STRUCTURE

#### Scene 1: Clinical Challenge – The Need for Diagnostic Transformation (0:00 – 0:20)
**Visual:**  
Show:  
● Doctors manually reviewing conflicting paper lab reports with frustration  
● Patients visiting multiple clinics with unexplained tremors and liver symptoms  
● Disconnected communication between liver specialists and neurologists  
● High-impact graphic showing the alarming 2-year average diagnostic delay  
● Hospital management concerned about diagnostic delays and patient risk  

**Voice Over:**  
"Welcome to NeuroMed AI, the Complete Clinical Decision Platform for Wilson’s Disease and Rare Metabolic Disorders.  
Every year, thousands of patients suffer irreversible liver and brain damage simply because rare diseases are difficult to identify. Traditional manual processes create challenges in maintaining diagnostic speed, cross-specialty coordination, and timely clinical decision-making."

**On-Screen Text:**  
Challenges in Traditional Diagnosis  
● Dangerous Diagnostic Delays (Up to 24 Months)  
● High Risk of Organ Misdiagnosis  
● Fragmented Multi-Organ Lab Reports  
● Lack of Standardized Decision Tools  

---

#### Scene 2: Introducing NeuroMed AI (0:20 – 0:45)
**Visual:**  
Show NeuroMed AI branding, executive clinical dashboard overview, and modern hospital-grade interface with intuitive navigation.

**Voice Over:**  
"NeuroMed AI is a complete digital solution that simplifies rare disease detection through clinical automation, real-time risk assessment, and easy-to-use medical workflows."

**Highlight:**  
✓ Complete Digital Clinical Workflow  
✓ Rapid Multi-Organ Risk Assessment  
✓ Automated International Scoring  
✓ Evidence-Based Treatment Protocols  
✓ Longitudinal Patient Tracking  
✓ Hospital MIS & Clinical Governance  

---

#### Scene 3: Centralized Patient and Clinical History Management (0:45 – 1:10)
**Visual:**  
Show:  
Application walkthrough:  
● Patient registration and digital intake  
● Comprehensive lab panel overview (liver enzymes, 24-hr urinary copper, ceruloplasmin)  
● Multi-organ symptom checklist (hepatic, neurological, ophthalmic)  
● Longitudinal timeline of patient tests and family screening history  

**Voice Over:**  
"NeuroMed AI enables healthcare organizations to maintain complete patient and diagnostic information through a centralized system. User-friendly workflows help clinical teams and hospital administrators manage records efficiently with improved diagnostic accuracy."

**Business Benefits:**  
✓ 360° Comprehensive Patient View  
✓ Faster Clinical Onboarding  
✓ Zero Overlooked Lab Findings  
✓ Unified Family & Genetic Tracking  

---

#### Scene 4: Diagnostic Assessment and Treatment Management (1:10 – 1:40)
**Visual:**  
Show:  
● One-click diagnostic assessment process  
● Automated Leipzig diagnostic score verification  
● Evidence-based treatment protocol (chelation therapy dosing and zinc maintenance)  
● Longitudinal recovery monitoring graph  

**Voice Over:**  
"From initial evaluation to long-term care, NeuroMed AI provides a complete disease management lifecycle. Clinical teams can easily evaluate patient risk and treatment pathways while medical directors gain real-time visibility into clinical outcomes."

**Highlight:**  
● Rapid Risk Stratification  
● Automated Leipzig Scoring  
● Evidence-Based Treatment Guidance  
● Longitudinal Recovery Monitoring  

---

#### Scene 5: Connected Bedside Operations and Mobile Solutions (1:40 – 2:05)
**Visual:**  
Show mobile and tablet application usage by physicians and clinical staff at the patient's bedside; seamless drag-and-drop lab report ingestion.

**Voice Over:**  
"Empower your clinical teams with digital tools that improve productivity and reduce manual effort. NeuroMed AI connects bedside medical teams with the central hospital system for faster and more accurate clinical decisions."

**Highlight:**  
● Mobile Bedside Access  
● Real-Time Record Synchronization  
● Instant Lab Report Ingestion  
● Cross-Specialty Doctor Collaboration  

---

#### Scene 6: Clinical Reports, Compliance and Security (2:05 – 2:35)
**Visual:**  
Show executive clinical dashboards, exportable diagnostic reports, audit trails, and healthcare security features.

**Voice Over:**  
"NeuroMed AI helps healthcare organizations maintain clinical transparency, international compliance, and better medical decision-making with powerful diagnostic reports and secure patient data management."

**Highlight:**  
✓ Real-Time Clinical MIS Reports  
✓ International Guideline Compliance (AASLD & EASL)  
✓ Role-Based Medical Access Control  
✓ Enterprise Health Data Security  
✓ Complete Clinical Audit Trails  

---

#### Scene 7: Integrations and Future-Ready Platform (2:35 – 2:50)
**Visual:**  
Show ecosystem diagram:  
NeuroMed AI Core Platform connected with:  
● Hospital Electronic Medical Records (EMR / EHR)  
● Laboratory Information Systems (LIS)  
● Automated Doctor & Patient Alerts  
● Secure Hospital Cloud & Mobile Applications  

**Voice Over:**  
"NeuroMed AI is designed to connect seamlessly with modern hospital systems through integrations such as Electronic Health Records, laboratory information systems, automated clinical alerts, and digital health networks."

---

#### Scene 8: Closing Call to Action (2:50 – 3:00)
**Visual:**  
Show:  
NeuroMed AI logo + official demo link and hospital partnership contact details  
● NeuroMed AI live platform in modern clinical setting  
● Confident medical team and thriving recovered patient  
● Hospital growth, safety, and clinical excellence badges  
● Contact information & consultation QR code  

**Voice Over:**  
"Transform your hospital’s rare disease care with NeuroMed AI. Move from manual, delayed processes to a complete digital ecosystem and empower your organization with smarter clinical intelligence."

**Text on Screen:**  
"NeuroMed AI – From First Lab Test to Lifesaving Care, Everything in One Intelligent Clinical Platform."

---

### VIDEO STYLE REQUIREMENTS
● Use clean healthcare corporate animation style.  
● Use modern healthtech visuals (deep slate blues, clean glassmorphism, clinical green accents).  
● Use smooth transitions between views.  
● Add professional, calm background music.  
● Use clear, authoritative English voice narration.  
● Maintain brand colors and high-end hospital appearance.  
● Avoid technical coding/AI jargon (no mentions of algorithms, code, or mathematics).  
● Focus strictly on business outcomes, hospital efficiency, and patient lives saved.  

---

### APPLICATION DEMO SECURITY REQUIREMENTS
When showing application screens:  
● Use only approved mock clinical demo data.  
● Never display real doctor usernames, passwords, API keys, or confidential hospital data.  
● Mask all sensitive patient fields using a post-production overlay.  
● Ensure no confidential information appears in screenshots, recordings, captions, voice narration, or final video output.  
● Review the complete video frame-by-frame before publishing.  

---

### OUTPUT TO GENERATE
1. Complete video storyboard  
2. Scene-wise visual instructions  
3. Voice-over script  
4. On-screen text captions  
5. AI video generation prompts for each scene  

---

### ROLE & PERSONA
Act as a **Healthcare SaaS Product Marketing Specialist and Medical Tech Video Director**. Create the video with the quality and storytelling style used by top enterprise healthcare software companies presenting digital transformation solutions to hospital boards and clinical directors.

---

### IMPORTANT CREATIVE PRINCIPLE
Do not make the video feel like a software manual.  
The viewer should experience this journey:  
**Problem → Frustration → NeuroMed AI → Simplicity → Control → Visibility → Confidence → Call to Action**  

The product interface should provide evidence of the solution.  
The story should sell the clinical and business outcome, not merely list features.  
The final impression should be:  
> *"NeuroMed AI makes complex rare disease diagnosis and clinical management simpler, more organized, and lifesaving."*

If any check fails, fix the problem and re-run the checks.

---

### EXPORT
- H.264 High profile, yuv420p, 30 fps, AAC 192 kbps at 48 kHz stereo, with the `-movflags +faststart` option.
- Deliverables: `NeuroMed_AI_Promo.mp4`, `neuromed_ai_promo.srt`, `narration_script.txt`, and the project source.
