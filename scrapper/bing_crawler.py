import os
import time
import random
from icrawler.builtin import BingImageCrawler

# 16 Pakistani Politicians with comprehensive search queries
# Primary queries (always used) + Backup queries (used if needed) + Expansion queries (all-out mode)
POLITICIANS = {
    "imran_khan": {
        "primary": ["Imran Khan Pakistan PM face", "Imran Khan PTI"],
        "backup": ["Imran Khan cricket", "Imran Khan speech Pakistan"],
        "expansion": [
            "Imran Khan close-up portrait headshot",
            "Imran Khan election rally 2024",
            "Imran Khan parliament speaking",
            "Imran Khan court appearance",
            "Imran Khan uniform formal photo",
            "عمران خان تصویر",
            "Imran Khan youth movement 2023",
        ],
    },
    "nawaz_sharif": {
        "primary": ["Nawaz Sharif Pakistan PM", "Nawaz Sharif PML-N"],
        "backup": ["Nawaz Sharif speech", "Nawaz Sharif rally"],
        "expansion": [
            "Nawaz Sharif close-up portrait",
            "Nawaz Sharif business meeting",
            "Nawaz Sharif parliament appearance",
            "Nawaz Sharif London residence",
            "Nawaz Sharif political gathering",
            "نواز شریف تصویر",
            "Nawaz Sharif 2022 2023 2024 photo",
            "Nawaz Sharif formal suit portrait",
        ],
    },
    "asif_ali_zardari": {
        "primary": ["Asif Ali Zardari Pakistan President", "Zardari PPP"],
        "backup": ["Asif Zardari speech", "Zardari Pakistan politics"],
        "expansion": [
            "Asif Ali Zardari official portrait",
            "Asif Ali Zardari office meeting",
            "Zardari press conference",
            "Asif Ali Zardari formal attire",
            "Zardari PPP rally event",
            "آصف علی زرداری",
            "Asif Zardari presidency photo",
            "Zardari diplomatic meeting",
        ],
    },
    "bilawal_bhutto": {
        "primary": ["Bilawal Bhutto Zardari Pakistan", "Bilawal PPP Chairman"],
        "backup": ["Bilawal Bhutto speech", "Bilawal Bhutto rally"],
        "expansion": [
            "Bilawal Bhutto close-up portrait",
            "Bilawal Bhutto foreign minister",
            "Bilawal Bhutto political event",
            "Bilawal Bhutto press briefing",
            "Bilawal Bhutto youth leader",
            "بلاول بھٹو",
            "Bilawal Bhutto parliament",
            "Bilawal Bhutto formal photo",
        ],
    },
    "shahbaz_sharif": {
        "primary": ["Shahbaz Sharif Pakistan PM", "Shehbaz Sharif PML-N"],
        "backup": ["Shahbaz Sharif Punjab", "Shehbaz Sharif speech"],
        "expansion": [
            "Shehbaz Sharif close-up portrait",
            "Shahbaz Sharif Prime Minister office",
            "Shehbaz Sharif infrastructure project",
            "Shahbaz Sharif parliament session",
            "Shehbaz Sharif formal government photo",
            "شاہباز شریف",
            "Shahbaz Sharif CM Punjab",
            "Shehbaz Sharif meeting officials",
        ],
    },
    "maryam_nawaz": {
        "primary": ["Maryam Nawaz Pakistan", "Maryam Nawaz PML-N"],
        "backup": ["Maryam Nawaz speech", "Maryam Nawaz rally"],
        "expansion": [
            "Maryam Nawaz close-up portrait",
            "Maryam Nawaz political campaign",
            "Maryam Nawaz Pakistan politics",
            "Maryam Nawaz press conference",
            "Maryam Nawaz public appearance",
            "مریم نواز",
            "Maryam Nawaz lady leader",
            "Maryam Nawaz formal photo",
        ],
    },
    "fazlur_rehman": {
        "primary": ["Fazlur Rehman Pakistan", "Maulana Fazlur Rehman JUI"],
        "backup": ["Fazal ur Rehman speech", "Maulana Fazal Pakistan", "Fazlur Rehman JUI-F chief"],
        "expansion": [
            "Maulana Fazlur Rehman close-up",
            "Fazlur Rehman JUI-F rally",
            "Maulana Fazal religious gathering",
            "Fazlur Rehman parliament speech",
            "Maulana Fazlur Rehman formal portrait",
            "فضل الرحمن",
            "Fazlur Rehman political event",
            "Maulana Fazal meeting officials",
        ],
    },
    "asfandyar_wali": {
        "primary": ["Asfandyar Wali Khan Pakistan", "Asfandyar Wali ANP"],
        "backup": ["Asfandyar Wali speech", "Asfandyar ANP leader"],
        "expansion": [
            "Asfandyar Wali Khan close-up portrait",
            "Asfandyar Wali ANP rally",
            "Asfandyar Khan Pashtun nationalist",
            "Asfandyar Wali parliament appearance",
            "Asfandyar Wali Bacha Khan anniversary",
            "اسفندیار والی خان",
            "Asfandyar Wali political gathering",
            "Asfandyar Khan formal government photo",
        ],
    },
    "altaf_hussain": {
        "primary": ["Altaf Hussain MQM Pakistan", "Altaf Hussain London"],
        "backup": ["Altaf Bhai MQM", "Altaf Hussain speech", "Altaf Hussain founder MQM"],
        "expansion": [
            "Altaf Hussain close-up portrait",
            "Altaf Hussain MQM founder",
            "Altaf Bhai rallies speeches",
            "Altaf Hussain Karachi MQM",
            "Altaf Hussain political gathering",
            "الطاف حسین",
            "Altaf Hussain archived photo",
            "MQM founder portrait",
        ],
    },
    "chaudhry_shujaat": {
        "primary": ["Chaudhry Shujaat Hussain Pakistan", "Shujaat Hussain PML-Q"],
        "backup": ["Chaudhry Shujaat speech", "Shujaat Hussain Pakistan politics"],
        "expansion": [
            "Chaudhry Shujaat Hussain close-up",
            "Shujaat Hussain PML-Q leader",
            "Chaudhry Shujaat parliament",
            "Shujaat Hussain Prime Minister",
            "Chaudhry Shujaat formal portrait",
            "چوہدری شجاعت",
            "Shujaat Hussain government photo",
            "Chaudhry Shujaat political rally",
        ],
    },
    "pervez_musharraf": {
        "primary": ["Pervez Musharraf Pakistan President", "General Musharraf"],
        "backup": ["Pervez Musharraf army", "Musharraf Pakistan military"],
        "expansion": [
            "General Pervez Musharraf close-up",
            "Pervez Musharraf military uniform",
            "General Musharraf formal portrait",
            "Pervez Musharraf President office",
            "Musharraf Pakistan 2001 2007 2008",
            "پرویز مشرف",
            "General Musharraf archive photo",
            "Pervez Musharraf government official",
        ],
    },
    "shehryar_afridi": {
        "primary": ["Shehryar Afridi Pakistan PTI", "Shehryar Khan Afridi"],
        "backup": ["Shehryar Afridi minister", "Shehryar Afridi narcotics", "Shehryar Afridi press conference"],
        "expansion": [
            "Shehryar Afridi close-up portrait",
            "Shehryar Khan Afridi government",
            "Shehryar Afridi anti-narcotics drive",
            "Shehryar Afridi parliament appearance",
            "Shehryar Afridi PTI politics",
            "شہریار افریدی",
            "Shehryar Afridi formal photo",
            "Shehryar Khan politician Pakistan",
        ],
    },
    "khawaja_asif": {
        "primary": ["Khawaja Asif Pakistan PML-N", "Khawaja Muhammad Asif"],
        "backup": ["Khawaja Asif minister", "Khawaja Asif speech"],
        "expansion": [
            "Khawaja Asif close-up portrait",
            "Khawaja Muhammad Asif Defence Minister",
            "Khawaja Asif parliament speech",
            "Khawaja Asif formal government photo",
            "Khawaja Asif PML-N politician",
            "خواجہ آصف",
            "Khawaja Asif office meeting",
            "Khawaja Muhammad Asif briefing",
        ],
    },
    "ahsan_iqbal": {
        "primary": ["Ahsan Iqbal Pakistan PML-N", "Ahsan Iqbal Minister"],
        "backup": ["Ahsan Iqbal planning", "Ahsan Iqbal speech"],
        "expansion": [
            "Ahsan Iqbal close-up portrait",
            "Ahsan Iqbal Planning Minister",
            "Ahsan Iqbal parliament appearance",
            "Ahsan Iqbal development project",
            "Ahsan Iqbal formal government photo",
            "احسن اقبال",
            "Ahsan Iqbal political event",
            "Ahsan Iqbal minister briefing",
        ],
    },
    "barrister_gohar": {
        "primary": ["Barrister Gohar Ali Khan PTI", "Gohar Ali Khan Pakistan"],
        "backup": ["Barrister Gohar PTI chairman", "Gohar Ali Khan lawyer"],
        "expansion": [
            "Barrister Gohar Ali Khan close-up",
            "Gohar Ali Khan PTI chairman",
            "Barrister Gohar parliament speech",
            "Gohar Ali Khan formal attire",
            "Barrister Gohar political gathering",
            "بیرسٹر گوہر",
            "Gohar Ali Khan official photo",
            "Barrister Gohar courtroom legal",
        ],
    },
    "ahmed_sharif_chaudhry": {
        "primary": ["Ahmed Sharif Chaudhry ISPR Pakistan", "DG ISPR Ahmed Sharif"],
        "backup": ["Ahmed Sharif ISPR briefing", "Lt Gen Ahmed Sharif Pakistan"],
        "expansion": [
            "Lt General Ahmed Sharif close-up",
            "Ahmed Sharif ISPR Director General",
            "Lt Gen Ahmed Sharif uniform",
            "Ahmed Sharif military briefing",
            "Ahmed Sharif Chaudhry formal photo",
            "احمد شریف چوہدری",
            "DG ISPR military official Pakistan",
            "Lt General Ahmed Sharif spokesman",
        ],
    },
}


def augment_generic_backups(politicians):
    generic_backups = ["official portrait", "press conference"]
    for key, data in politicians.items():
        display_name = key.replace("_", " ").title()
        backups = data.get("backup", [])
        for generic_query in generic_backups:
            query = f"{display_name} {generic_query}"
            if query not in backups:
                backups.append(query)
        data["backup"] = backups


augment_generic_backups(POLITICIANS)


def crawl_images_adaptive(name, query_dict, max_per_query=200, max_first_query=300, target_raw=250):
    """Smart adaptive image collection with automatic backup + expansion queries"""
    out_dir = f"data/raw/{name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📸 Collecting: {name}")
    print(f"   Target: {target_raw} raw images (before filtering)")

    def count_images():
        if not os.path.exists(out_dir):
            return 0
        return len([f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def crawl_query(query, num_images, attempts=5):
        for attempt in range(1, attempts + 1):
            try:
                delay = random.uniform(0.5, 2.0)
                time.sleep(delay)

                crawler = BingImageCrawler(
                    storage={"root_dir": out_dir},
                    downloader_threads=4,
                )
                crawler.crawl(keyword=query, max_num=num_images)
                return True
            except Exception as exc:
                print(f"  ✗ Bing attempt {attempt} for '{query}' failed: {exc}")
                if attempt < attempts:
                    backoff = 2 ** attempt
                    print(f"    ↻ retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    print(f"    ✗ all attempts failed for '{query}'")
                    return False

    print(f"  📍 Phase 1: Primary queries (first-pass max {max_first_query})")
    for query in query_dict.get("primary", []):
        crawl_query(query, num_images=max_first_query)
        current_count = count_images()
        print(f"     Current: {current_count} images")
        if current_count >= target_raw:
            print("  ✅ Target reached with primary queries!")
            return
        time.sleep(1)

    current_count = count_images()
    if current_count < target_raw:
        print(f"  ⚠️  Only {current_count} images (need {target_raw})")
        print("  📍 Phase 2: Using backup queries...")
        for query in query_dict.get("backup", []):
            crawl_query(query, num_images=max_per_query)
            current_count = count_images()
            print(f"     Current: {current_count} images")
            if current_count >= target_raw:
                print("  ✅ Target reached with backup queries!")
                return
            time.sleep(1)

    current_count = count_images()
    if current_count < target_raw:
        expansion_queries = query_dict.get("expansion", [])
        if expansion_queries:
            print(f"  ⚠️  Only {current_count} images (need {target_raw})")
            print("  📍 Phase 3: Using expansion queries (all-out mode)...")
            for query in expansion_queries:
                crawl_query(query, num_images=max_per_query)
                current_count = count_images()
                print(f"     Current: {current_count} images")
                if current_count >= target_raw:
                    print("  ✅ Target reached with expansion queries!")
                    return
                time.sleep(1)

    final_count = count_images()
    if final_count < target_raw:
        print(f"  ⚠️  Final: {final_count} images (below target of {target_raw})")
        print("  💡 Will continue with available images")
    else:
        print(f"  ✅ Final: {final_count} images collected!")


if __name__ == "__main__":
    print("\nStarting aggressive web scraping with expanded parameters...")
    print("Strategy:")
    print("   - Collect: ~300 images per politician first pass (expanded targets)")
    print("   - Face alignment: MTCNN + 336x336 (if enabled)")
    print("   - Deduplication: pHash distance <= 5 (if enabled)")
    print("   - Target: 300 raw images per politician (before filtering)")
    print("   - Augmentation: offline only for small classes")
    print("   - Final training: Expected 150-200 images per class after split + augmentation")
    print("   - SMART MODE: Automatically uses backup + expansion queries if needed")
    print("   - PARALLELISM: 8 downloader threads per crawler")
    print("   - RESILIENCE: 5 retry attempts with exponential backoff + polite delays\n")

    for politician_name, query_dict in POLITICIANS.items():
        crawl_images_adaptive(
            politician_name,
            query_dict,
            max_per_query=200,
            max_first_query=300,
            target_raw=250,
        )
