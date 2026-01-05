#!/usr/bin/env python3
"""Automated download of historical OMI data using Playwright.

This script:
1. Opens Agenzia delle Entrate portal
2. Pauses for manual SPID/CIE login
3. Navigates to OMI service and downloads all historical data
4. Tracks progress for resume capability

Usage:
    # First install playwright and browsers
    uv pip install playwright
    playwright install chromium

    # Run the script
    python scripts/download_omi_historical.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("ERROR: Playwright not installed.")
    print("Run: uv pip install 'italian-real-estate-risk[scraping]'")
    print("Then: playwright install chromium")
    sys.exit(1)


# Configuration
PORTAL_URL = "https://telematici.agenziaentrate.gov.it/Main/index.jsp"
OMI_SERVICE_PATH = "Servizi ipotecari e catastali, Osservatorio Mercato Immobiliare"
DOWNLOAD_DIR = Path(__file__).parent.parent / "data" / "raw" / "omi" / "historical"
STATE_FILE = Path(__file__).parent.parent / "data" / "raw" / "omi" / ".download_state.json"

# All semesters to download (2004-2024)
ALL_SEMESTERS = []
for year in range(2004, 2025):
    ALL_SEMESTERS.append((year, 1))
    ALL_SEMESTERS.append((year, 2))


def load_state() -> dict:
    """Load download progress state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "last_run": None}


def save_state(state: dict):
    """Save download progress state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_pending_semesters(state: dict) -> list[tuple[int, int]]:
    """Get list of semesters not yet downloaded."""
    downloaded = set(tuple(x) for x in state.get("downloaded", []))
    return [s for s in ALL_SEMESTERS if s not in downloaded]


def download_omi_data():
    """Main download function with Playwright automation."""
    # Ensure download directory exists
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load state
    state = load_state()
    pending = get_pending_semesters(state)

    if not pending:
        print("All semesters already downloaded!")
        print(f"Downloaded: {len(state['downloaded'])} semesters")
        return

    print(f"Pending downloads: {len(pending)} semesters")
    print(f"Already downloaded: {len(state['downloaded'])} semesters")
    print()

    with sync_playwright() as p:
        # Launch browser with visible window
        browser = p.chromium.launch(
            headless=False,
            downloads_path=str(DOWNLOAD_DIR),
        )

        context = browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
        )

        page = context.new_page()

        print("=" * 60)
        print("STEP 1: Opening Agenzia delle Entrate portal...")
        print("=" * 60)
        page.goto(PORTAL_URL)

        print()
        print("=" * 60)
        print("STEP 2: MANUAL LOGIN REQUIRED")
        print("=" * 60)
        print()
        print("Please complete login in the browser window:")
        print("  1. Click 'Accedi' (Login)")
        print("  2. Choose SPID, CIE, or credentials")
        print("  3. Complete the authentication")
        print()
        print("The script will resume automatically after login.")
        print("Press Ctrl+C to abort at any time.")
        print()

        # Pause and let user login - this opens Playwright Inspector
        page.pause()

        print()
        print("=" * 60)
        print("STEP 3: Navigating to OMI service...")
        print("=" * 60)

        # After login, navigate to OMI service
        # The exact navigation depends on the portal structure
        # We'll try multiple approaches

        try:
            # Look for OMI link
            omi_link = page.locator("text=Osservatorio Mercato Immobiliare").first
            if omi_link.is_visible():
                omi_link.click()
                page.wait_for_load_state("networkidle")
        except Exception as e:
            print(f"Navigation hint: {e}")
            print("If automatic navigation fails, please navigate manually to:")
            print("  'Servizi ipotecari e catastali' > 'Forniture OMI'")
            page.pause()

        print()
        print("=" * 60)
        print("STEP 4: Downloading historical data...")
        print("=" * 60)

        # Download each pending semester
        for year, semester in pending:
            print(f"\nDownloading {year} S{semester}...")

            try:
                # This is where we'd navigate to download the specific semester
                # The exact selectors depend on the portal's UI

                # Try to find year/semester selection
                # Note: You may need to adjust these selectors based on actual portal UI

                # Select year
                year_select = page.locator("select[name*='anno'], select[id*='anno']").first
                if year_select.is_visible():
                    year_select.select_option(str(year))

                # Select semester
                sem_select = page.locator("select[name*='semestre'], select[id*='semestre']").first
                if sem_select.is_visible():
                    sem_select.select_option(str(semester))

                # Select national territory option
                territory = page.locator("input[value*='nazionale'], label:has-text('Intero territorio')").first
                if territory.is_visible():
                    territory.click()

                # Click download/submit
                download_btn = page.locator("button:has-text('Scarica'), input[type='submit'], button:has-text('Download')").first

                # Handle download
                with page.expect_download(timeout=120000) as download_info:
                    download_btn.click()

                download = download_info.value

                # Save with standardized name
                filename = f"quotazioni_{year}S{semester}.csv"
                filepath = DOWNLOAD_DIR / filename
                download.save_as(filepath)

                print(f"  ✓ Saved: {filename}")

                # Update state
                state["downloaded"].append([year, semester])
                save_state(state)

                # Brief pause between downloads to be nice to the server
                time.sleep(2)

            except PlaywrightTimeout:
                print(f"  ✗ Timeout downloading {year} S{semester}")
                state["failed"].append([year, semester])
                save_state(state)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                print()
                print("Automatic download failed. Opening pause for manual download.")
                print(f"Please download {year} S{semester} manually, then press Resume.")
                page.pause()

                # Check if file was downloaded manually
                expected_file = DOWNLOAD_DIR / f"quotazioni_{year}S{semester}.csv"
                if expected_file.exists():
                    print(f"  ✓ Manual download detected: {expected_file.name}")
                    state["downloaded"].append([year, semester])
                    save_state(state)

        print()
        print("=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"Downloaded: {len(state['downloaded'])} semesters")
        print(f"Failed: {len(state['failed'])} semesters")
        print()
        print(f"Files saved to: {DOWNLOAD_DIR}")
        print()
        print("Next step: Run the ingestion script:")
        print("  python scripts/ingest_historical_omi.py")

        browser.close()


def check_downloads():
    """Check what's been downloaded."""
    state = load_state()

    print("Download Status")
    print("=" * 40)
    print(f"Downloaded: {len(state['downloaded'])} semesters")
    print(f"Failed: {len(state['failed'])} semesters")
    print(f"Pending: {len(get_pending_semesters(state))} semesters")
    print()

    # Check actual files
    if DOWNLOAD_DIR.exists():
        files = list(DOWNLOAD_DIR.glob("*.csv"))
        print(f"CSV files in directory: {len(files)}")
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")


def reset_state():
    """Reset download state (for re-downloading)."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print("State reset. All semesters will be re-downloaded.")
    else:
        print("No state file to reset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download historical OMI data")
    parser.add_argument("--check", action="store_true", help="Check download status")
    parser.add_argument("--reset", action="store_true", help="Reset download state")
    args = parser.parse_args()

    if args.check:
        check_downloads()
    elif args.reset:
        reset_state()
    else:
        download_omi_data()
