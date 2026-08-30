"""Game HWND discovery must survive renamed clients without matching browsers."""
from maple_analyzer.capture import GameWindowCapture


class _Win32Gui:
    def __init__(self, *, title, size=(2560, 1440)):
        self.title = title
        self.size = size

    def GetWindowText(self, _hwnd):
        return self.title

    def GetClientRect(self, _hwnd):
        return 0, 0, self.size[0], self.size[1]


def _capture(*, title, owner, size=(2560, 1440)):
    capture = GameWindowCapture.__new__(GameWindowCapture)
    capture._win32gui = _Win32Gui(title=title, size=size)
    capture._title_substring = "新楓之谷"
    capture._title_tokens = ("新楓之谷", "MapleStory")
    capture._process_name = "maplestory"
    capture._owning_process_name = lambda _hwnd: owner
    return capture


def test_renamed_game_process_keeps_native_chinese_title_discoverable():
    capture = _capture(title="新楓之谷：經典版", owner="renamed-client.exe")
    assert capture._is_match(100) is True


def test_english_game_title_is_discoverable_on_another_launcher():
    capture = _capture(title="MapleStory", owner="client64.exe")
    assert capture._is_match(100) is True


def test_browser_or_small_title_match_is_not_treated_as_the_game():
    browser = _capture(
        title="新楓之谷攻略 - Google Chrome",
        owner="chrome.exe",
    )
    assert browser._is_match(100) is False

    small = _capture(
        title="新楓之谷：經典版",
        owner="renamed-client.exe",
        size=(400, 300),
    )
    assert small._is_match(100) is False
