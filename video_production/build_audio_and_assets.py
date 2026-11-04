"""
NeuroMed AI — 1-Minute Marketing Audio Generator
Uses edge-tts to generate professional marketing narration with en-IN-NeerjaNeural.
"""
import asyncio
from pathlib import Path

VOICE = "en-IN-NeerjaNeural"
ALT_VOICE = "en-IN-PrabhatNeural"
RATE = "-8%"
VOLUME = "+0%"

SCENES = [
    {
        "id": "scene_1",
        "title": "Clinical Challenge",
        "text": "Wilson's disease is a treatable genetic disorder, yet subtle symptoms cause a dangerous two-year diagnostic delay—risking irreversible liver failure and permanent neurological damage.",
        "duration": 12.0
    },
    {
        "id": "scene_2",
        "title": "Introducing NeuroMed AI",
        "text": "Meet NeuroMed AI: the complete clinical decision platform that empowers hospital teams to detect, diagnose, and manage Wilson’s disease in minutes, not months.",
        "duration": 12.0
    },
    {
        "id": "scene_3",
        "title": "Intelligent Clinical Assessment",
        "text": "With one click, NeuroMed AI evaluates multi-organ lab panels, automates international Leipzig scoring, and delivers instant risk stratification—eliminating diagnostic guesswork for clinical teams.",
        "duration": 12.0
    },
    {
        "id": "scene_4",
        "title": "Guideline-Backed Care",
        "text": "Physicians receive evidence-based treatment protocols grounded in top international clinical guidelines, while seamlessly connecting hepatologists, neurologists, and pathology labs for longitudinal patient monitoring.",
        "duration": 12.0
    },
    {
        "id": "scene_5",
        "title": "Closing Call to Action",
        "text": "Protect your patients from fatal copper toxicity. Transform your hospital’s rare disease care with NeuroMed AI. Schedule your clinical demonstration today.",
        "duration": 12.0
    }
]

async def generate_narration():
    try:
        import edge_tts
    except ImportError:
        print("[!] edge-tts not installed. Install with: pip install edge-tts")
        return

    output_dir = Path(__file__).parent / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_text = " ".join([s["text"] for s in SCENES])
    full_output = output_dir / "marketing_60s_narration.mp3"
    
    print(f"[*] Generating 60s marketing narration with {VOICE}...")
    communicate = edge_tts.Communicate(full_text, voice=VOICE, rate=RATE, volume=VOLUME)
    await communicate.save(str(full_output))
    print(f"[+] Full 60s narration saved to: {full_output}")

    for s in SCENES:
        scene_output = output_dir / f"{s['id']}_marketing.mp3"
        comm = edge_tts.Communicate(s["text"], voice=VOICE, rate=RATE, volume=VOLUME)
        await comm.save(str(scene_output))
        print(f"[+] Saved {s['id']}: {s['duration']}s")

if __name__ == "__main__":
    asyncio.run(generate_narration())
