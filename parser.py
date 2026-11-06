import argparse
import json
import os
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

from config import (
    EDGE_DRIVER_PATH,
    MAIN_URL,
    OUTPUT_DIR,
    PARSED_URLS_FILE,
    PARSER_HEADLESS,
    PARSER_PAGE_TIMEOUT,
)


# ─────────────── Реестр URL ───────────────────────
def load_parsed_urls() -> set:
    if not os.path.exists(PARSED_URLS_FILE):
        return set()
    try:
        with open(PARSED_URLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Не удалось прочитать реестр URL: {exc}")
    return set()


def save_parsed_urls(urls: set):
    """Атомарно сохраняет реестр URL, чтобы не потерять его при сбое."""
    tmp_path = f"{PARSED_URLS_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(urls), f, ensure_ascii=False, indent=0)
    os.replace(tmp_path, PARSED_URLS_FILE)


# ─────────────── Драйвер ─────────────────────────
def create_driver(headless: bool | None = None) -> webdriver.Edge:
    opts = webdriver.EdgeOptions()
    opts.page_load_strategy = "eager"
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--log-level=3")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-extensions")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    use_headless = PARSER_HEADLESS if headless is None else headless
    if use_headless:
        opts.add_argument("--headless=new")

    # Если драйвер не задан или не найден, Selenium Manager подберёт его сам.
    if EDGE_DRIVER_PATH and os.path.exists(EDGE_DRIVER_PATH):
        service = Service(EDGE_DRIVER_PATH)
        driver = webdriver.Edge(options=opts, service=service)
    else:
        driver = webdriver.Edge(options=opts)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"},
    )

    driver.set_page_load_timeout(PARSER_PAGE_TIMEOUT)
    driver.implicitly_wait(5)
    return driver


# ───────────── Загрузка страницы ──────────────────
def safe_get(driver, url: str, wait_after: float = 1.5):
    """Загружает страницу. При таймауте — останавливает и продолжает."""
    try:
        driver.get(url)
    except Exception:
        try:
            driver.execute_cdp_cmd("Page.stopLoading", {})
        except Exception:
            pass
    time.sleep(wait_after)


# ─────────────── Сбор ссылок ──────────────────────
def take_main_links(driver, url):
    safe_get(driver, url, wait_after=3)

    li_tags = driver.find_elements(By.CSS_SELECTOR, "li.wos")

    links = []
    for li in li_tags:
        for a in li.find_elements(By.TAG_NAME, "a"):
            href = a.get_attribute("href")
            if href:
                links.append(href)

    print(f"Main page: {len(links)} links")
    return links


def take_sub_main_links(driver, page_links, max_categories: int | None = None):
    sub = []
    selected = page_links[:max_categories] if max_categories else page_links
    for i, link in enumerate(selected, 1):
        safe_get(driver, link, wait_after=1)

        for li in driver.find_elements(By.CSS_SELECTOR, "li.wos"):
            for a in li.find_elements(By.TAG_NAME, "a"):
                href = a.get_attribute("href")
                if href:
                    sub.append(href)

    print(f"Subcategories: {len(sub)} links")
    return sub


def get_final_document_links(
    driver,
    category_links,
    already_parsed: set,
    max_pages: int | None = None,
    limit: int | None = None,
):
    final = []
    for ci, link in enumerate(category_links, 1):
        page_num = 1
        while True:
            if max_pages is not None and page_num > max_pages:
                break
            if limit is not None and len(final) >= limit:
                break

            page_url = f"{link}?page={page_num}"
            safe_get(driver, page_url, wait_after=1)
            try:
                results_ul = driver.find_element(By.CSS_SELECTOR, ".results")
                a_tags = results_ul.find_elements(By.TAG_NAME, "a")
                if not a_tags:
                    break

                added = 0
                for el in a_tags:
                    txt = el.text.strip()
                    if txt.startswith(("Решение", "Определение")):
                        href = el.get_attribute("href")
                        if href and href not in already_parsed and href not in final:
                            final.append(href)
                            added += 1
                            if limit is not None and len(final) >= limit:
                                break
                if added == 0:
                    break
                page_num += 1
            except NoSuchElementException:
                break

        if ci % 10 == 0:
            print(f"  Category [{ci}/{len(category_links)}], docs: {len(final)}")
        if limit is not None and len(final) >= limit:
            break

    print(f"New documents: {len(final)}")
    return final


# ─────────────── Парсинг текста ───────────────────
def parse_text_from_html(driver, url: str) -> str | None:
    """Метод нарезки HTML по маркерам."""
    safe_get(driver, url, wait_after=1.5)

    try:
        src = driver.page_source
    except Exception:
        return None

    start_marker = '<hr class="hr-h1">'
    end_marker = "<h3>Суд:</h3>"
    alt_end = '<div class="adv_inside_text">'

    si = src.find(start_marker)
    if si == -1:
        return None

    sp = si + len(start_marker)
    ei = src.find(end_marker, sp)
    if ei == -1:
        ei = src.find(alt_end, sp)
    if ei == -1:
        return None

    soup = BeautifulSoup(src[sp:ei], "lxml")
    text = soup.get_text(separator="\n", strip=True)

    if len(text) < 100:
        return None

    return text


def process_single_document(driver, url: str, parsed_urls: set) -> bool:
    slug = url.strip("/").split("/")[-1]
    if not slug:
        return False
    txt_path = os.path.join(OUTPUT_DIR, f"{slug}.txt")

    text = parse_text_from_html(driver, url)
    if not text:
        return False

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    parsed_urls.add(url)
    return True


# ─────────────── Основной конвейер ────────────────
def run_full_pipeline(
    limit: int | None = None,
    max_categories: int | None = None,
    max_pages: int | None = None,
    headless: bool | None = None,
) -> dict:
    parsed_urls = load_parsed_urls()
    driver = create_driver(headless=headless)

    success = 0
    fail = 0
    try:
        main_links = take_main_links(driver, MAIN_URL)
        sub_links = take_sub_main_links(driver, main_links, max_categories=max_categories)
        doc_links = get_final_document_links(
            driver,
            sub_links,
            parsed_urls,
            max_pages=max_pages,
            limit=limit,
        )

        print(f"Found {len(doc_links)} new documents")

        consecutive_fails = 0

        for i, url in enumerate(doc_links, 1):
            try:
                ok = process_single_document(driver, url, parsed_urls)
            except Exception as e:
                print(f"  [{i}] Error: {e}")
                ok = False
                consecutive_fails += 1

                if consecutive_fails >= 10:
                    print("  Recreating browser...")
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    time.sleep(3)
                    driver = create_driver(headless=headless)
                    consecutive_fails = 0
                continue

            if ok:
                success += 1
                consecutive_fails = 0
            else:
                fail += 1
                consecutive_fails += 1

            if i % 50 == 0:
                print(f"  [{i}/{len(doc_links)}] {success} ok / {fail} fail")

            if i % 10 == 0:
                save_parsed_urls(parsed_urls)

            if consecutive_fails >= 10:
                print("  Recreating browser...")
                try:
                    driver.quit()
                except Exception:
                    pass
                time.sleep(3)
                driver = create_driver(headless=headless)
                consecutive_fails = 0

    finally:
        save_parsed_urls(parsed_urls)
        try:
            driver.quit()
        except Exception:
            pass

    print(f"Done: {success} ok / {fail} fail")
    return {"success": success, "failed": fail}


if __name__ == "__main__":
    parser_args = argparse.ArgumentParser(
        description="Сбор судебных решений с sudact.ru"
    )
    parser_args.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимальное число новых документов",
    )
    parser_args.add_argument(
        "--max-categories",
        type=int,
        default=None,
        help="Максимальное число категорий верхнего уровня",
    )
    parser_args.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Максимальное число страниц пагинации в каждой подкатегории",
    )
    parser_args.add_argument(
        "--headless",
        action="store_true",
        help="Запустить браузер без графического интерфейса",
    )
    args = parser_args.parse_args()
    run_full_pipeline(
        limit=args.limit,
        max_categories=args.max_categories,
        max_pages=args.max_pages,
        headless=True if args.headless else None,
    )
