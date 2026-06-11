from pathlib import Path

from playwright.sync_api import sync_playwright

auth_dir = Path("playwright", ".auth")
auth_dir.mkdir(parents=True, exist_ok=True)


def save_magnit_session():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://magnit.ru", timeout=0)

        input("Press Enter to close the browser and save the session...")

        context.storage_state(path=auth_dir / "magnit.json")
        browser.close()

        print("Session saved to", auth_dir / "magnit.json")


def test_magnit_session():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=auth_dir / "magnit.json")
        page = context.new_page()
        page.goto("https://magnit.ru/")

        input("Press Enter to close the browser...")

        browser.close()

        print("Session tested successfully")


def main():
    x = input("1 - Save Magnit session\n2 - Test Magnit session\nChoose an option: ")
    if x == "1":
        save_magnit_session()
    elif x == "2":
        test_magnit_session()
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
