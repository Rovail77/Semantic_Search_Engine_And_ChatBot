from selenium.common.exceptions import NoSuchElementException

import parser as pr


class FakeAnchor:
    def __init__(self, href="", text=""):
        self._href = href
        self.text = text

    def get_attribute(self, name):
        return self._href if name == "href" else ""


class FakeContainer:
    def __init__(self, anchors):
        self._anchors = anchors

    def find_elements(self, by=None, value=None):
        return self._anchors


class FakeDriver:
    def __init__(self, links=None, pages=None):
        self.links = links or []
        self.pages = pages or {}
        self.current_url = ""
        self.quit_called = False

    def find_elements(self, by=None, value=None):
        return self.links

    def find_element(self, by=None, value=None):
        page = self.current_url
        if page in self.pages:
            return self.pages[page]
        raise NoSuchElementException()

    def get(self, url):
        self.current_url = url

    def execute_cdp_cmd(self, *args, **kwargs):
        return {}

    def set_page_load_timeout(self, timeout):
        pass

    def implicitly_wait(self, timeout):
        pass

    def quit(self):
        self.quit_called = True


def test_create_driver_without_service(monkeypatch):
    created = {}

    class Options:
        page_load_strategy = None

        def add_argument(self, *args):
            pass

        def add_experimental_option(self, *args):
            pass

    class FakeEdge:
        def __init__(self, options=None, service=None):
            created["driver"] = self
            self.cdp = []

        def execute_cdp_cmd(self, *args):
            self.cdp.append(args)

        def set_page_load_timeout(self, timeout):
            created["timeout"] = timeout

        def implicitly_wait(self, timeout):
            created["implicit"] = timeout

    monkeypatch.setattr(pr.webdriver, "EdgeOptions", lambda: Options())
    monkeypatch.setattr(pr.webdriver, "Edge", FakeEdge)
    monkeypatch.setattr(pr, "EDGE_DRIVER_PATH", "")
    monkeypatch.setattr(pr, "PARSER_HEADLESS", True)

    driver = pr.create_driver()

    assert isinstance(driver, FakeEdge)
    assert created["timeout"] == pr.PARSER_PAGE_TIMEOUT
    assert driver.cdp


def test_create_driver_with_existing_service(monkeypatch, tmp_path):
    driver_path = tmp_path / "msedgedriver.exe"
    driver_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(pr, "EDGE_DRIVER_PATH", str(driver_path))
    monkeypatch.setattr(pr.webdriver, "EdgeOptions", lambda: type(
        "O", (), {"add_argument": lambda *a: None, "add_experimental_option": lambda *a: None}
    )())
    monkeypatch.setattr(pr.webdriver, "Edge", lambda options=None, service=None: FakeDriver())
    monkeypatch.setattr(pr, "Service", lambda path: ("service", path))

    driver = pr.create_driver(headless=False)
    assert isinstance(driver, FakeDriver)


def test_take_main_links(monkeypatch):
    monkeypatch.setattr(pr, "safe_get", lambda *a, **k: None)
    driver = FakeDriver(
        links=[FakeContainer([FakeAnchor("https://a"), FakeAnchor("")])]
    )
    assert pr.take_main_links(driver, "https://main") == ["https://a"]


def test_take_sub_main_links_respects_category_limit(monkeypatch):
    visited = []
    monkeypatch.setattr(pr, "safe_get", lambda driver, url, **k: visited.append(url))
    driver = FakeDriver(links=[FakeContainer([FakeAnchor("https://doc")])])

    links = pr.take_sub_main_links(driver, ["c1", "c2", "c3"], max_categories=2)

    assert links == ["https://doc", "https://doc"]
    assert visited == ["c1", "c2"]


def test_get_final_document_links_paginates_until_empty(monkeypatch):
    monkeypatch.setattr(
        pr, "safe_get", lambda driver, url, **k: setattr(driver, "current_url", url)
    )
    page1 = FakeContainer(
        [FakeAnchor("https://d1", "Решение по делу"), FakeAnchor("https://x", "Не решение")]
    )
    page2 = FakeContainer([])
    driver = FakeDriver(pages={"cat?page=1": page1, "cat?page=2": page2})

    links = pr.get_final_document_links(driver, ["cat"], set(), max_pages=5)

    assert links == ["https://d1"]


def test_get_final_document_links_skips_already_parsed(monkeypatch):
    monkeypatch.setattr(
        pr, "safe_get", lambda driver, url, **k: setattr(driver, "current_url", url)
    )
    page1 = FakeContainer([FakeAnchor("https://d1", "Решение")])
    driver = FakeDriver(pages={"cat?page=1": page1})

    links = pr.get_final_document_links(driver, ["cat"], {"https://d1"}, max_pages=1)
    assert links == []


def test_get_final_document_links_respects_limit(monkeypatch):
    monkeypatch.setattr(
        pr, "safe_get", lambda driver, url, **k: setattr(driver, "current_url", url)
    )
    page1 = FakeContainer(
        [FakeAnchor("https://d1", "Решение"), FakeAnchor("https://d2", "Определение")]
    )
    driver = FakeDriver(pages={"cat?page=1": page1})

    assert pr.get_final_document_links(driver, ["cat"], set(), limit=1) == ["https://d1"]


def test_run_full_pipeline_counts_results(monkeypatch, tmp_path):
    driver = FakeDriver()
    monkeypatch.setattr(pr, "create_driver", lambda headless=None: driver)
    monkeypatch.setattr(pr, "take_main_links", lambda *a: ["cat"])
    monkeypatch.setattr(pr, "take_sub_main_links", lambda *a, **k: ["sub"])
    monkeypatch.setattr(pr, "get_final_document_links", lambda *a, **k: ["u1", "u2"])
    monkeypatch.setattr(pr, "process_single_document", lambda d, u, s: u == "u1")
    monkeypatch.setattr(pr, "save_parsed_urls", lambda urls: None)

    result = pr.run_full_pipeline(limit=2)

    assert result == {"success": 1, "failed": 1}
    assert driver.quit_called


def test_run_full_pipeline_recreates_driver_after_errors(monkeypatch):
    drivers = []

    def make_driver(headless=None):
        d = FakeDriver()
        drivers.append(d)
        return d

    def boom(driver, url, parsed):
        raise RuntimeError("page crashed")

    monkeypatch.setattr(pr, "create_driver", make_driver)
    monkeypatch.setattr(pr, "take_main_links", lambda *a: ["cat"])
    monkeypatch.setattr(pr, "take_sub_main_links", lambda *a, **k: ["sub"])
    monkeypatch.setattr(pr, "get_final_document_links", lambda *a, **k: [f"u{i}" for i in range(11)])
    monkeypatch.setattr(pr, "process_single_document", boom)
    monkeypatch.setattr(pr, "save_parsed_urls", lambda urls: None)
    monkeypatch.setattr(pr.time, "sleep", lambda _s: None)

    result = pr.run_full_pipeline()

    assert result["failed"] == 11
    assert len(drivers) >= 2


def test_create_driver_supports_chrome(monkeypatch):
    created = {}

    class Options:
        def add_argument(self, *args):
            pass

        def add_experimental_option(self, *args):
            pass

    class FakeChrome:
        def __init__(self, options=None, service=None):
            created["driver"] = self

        def execute_cdp_cmd(self, *args):
            pass

        def set_page_load_timeout(self, timeout):
            pass

        def implicitly_wait(self, timeout):
            pass

    monkeypatch.setattr(pr, "PARSER_BROWSER", "chrome")
    monkeypatch.setattr(pr, "CHROME_DRIVER_PATH", "")
    monkeypatch.setattr(pr.webdriver, "ChromeOptions", lambda: Options())
    monkeypatch.setattr(pr.webdriver, "Chrome", FakeChrome)

    driver = pr.create_driver(headless=True)

    assert isinstance(driver, FakeChrome)
