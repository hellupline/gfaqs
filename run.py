#!/usr/bin/env python3

import hashlib
import io
import json
import logging
import mimetypes
import zipfile

from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.parse import urlunparse

import click
import magic
import requests

from pyquery import PyQuery
from tenacity import after_log
from tenacity import before_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_fixed
from tqdm import tqdm


SYSTEMS = [
    "gameboy",
    "gba",
    "gbc",
    "snes",
]

cache_path = Path("cache")
game_data_path = Path("data/games")
guides_path = Path("guides")
logs_path = Path("logs")
logs_path.mkdir(parents=True, exist_ok=True)

log_format = "[%(asctime)s:%(levelname)s] %(message)s"
log_datefmt = "%Y-%m-%dT%H:%M:%S%z"
log_filename = logs_path / "{:%Y-%m-%dT%H:%M:%S}.log".format(datetime.now())
logging.basicConfig(format=log_format, datefmt=log_datefmt, filename=log_filename)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

headers = {
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/112.0.0.0 Safari/537.36"
    ),
    "accept": "text/html",
    "accept-language": "en-US,en;q=0.8",
}
session = requests.Session()
session.headers.update(headers)

mime_database = magic.Magic(mime=True)


class CacheMiss(Exception):
    pass


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


@dataclass()
class Platform:
    url: str = field(repr=False)
    name: str


@dataclass()
class Genre:
    url: str = field(repr=False)
    name: str


@dataclass()
class Developer:
    url: str = field(repr=False)
    name: str


@dataclass()
class Publisher:
    url: str = field(repr=False)
    name: str


@dataclass()
class Franchise:
    url: str = field(repr=False)
    name: str


@dataclass()
class Release:
    title: str
    region: str
    publisher: Publisher
    product_id: str
    release_date: Union[date, str]


@dataclass()
class GuideAuthor:
    url: str = field(repr=False)
    name: str


@dataclass()
class _Guide:
    platform: str
    url: str = field(repr=False)
    name: str
    author: GuideAuthor
    flairs: list[str] = field(repr=False)
    notes: str = field(repr=False)
    status: str = field(repr=False)
    key: str = field(repr=False, init=False)


class Guide(_Guide):
    platform: str
    url: str = field(repr=False)
    name: str
    author: GuideAuthor
    flairs: list[str] = field(repr=False)
    notes: str = field(repr=False)
    status: str = field(repr=False)

    @cached_property
    def key(self) -> str:  # type: ignore
        return get_cache_key(self.url)


@dataclass()
class GuideSection:
    name: str
    guides: list[Guide]


@dataclass()
class GuideGame:
    name: Union[str, None]
    sections: list[GuideSection]


@dataclass()
class Game:
    url: str = field(repr=False)
    slug: str
    name: str
    description: str = field(repr=False)
    platform: list[Platform] = field(default_factory=list)
    genre: list[Genre] = field(default_factory=list)
    developer: list[Developer] = field(default_factory=list)
    publisher: list[Publisher] = field(default_factory=list)
    franchises: list[Franchise] = field(default_factory=list)
    alternative_names: list[str] = field(default_factory=list)
    releases: list[Release] = field(default_factory=list)
    guides: list[GuideGame] = field(default_factory=list)


@click.group(invoke_without_command=True)
def cli() -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    game_data_path.mkdir(parents=True, exist_ok=True)
    guides_path.mkdir(parents=True, exist_ok=True)


@cli.command("download")
@click.option("--system-name", "system_names", type=click.Choice(SYSTEMS), multiple=True, required=True)
@click.option("--game-name", "game_names", type=click.STRING, multiple=True, required=True)
def cli_download(system_names: list[str], game_names: list[str]) -> None:
    for system_name in system_names:
        system_url = f"https://gamefaqs.gamespot.com/{system_name}/category/999-all"
        urls = sorted(get_games_urls(system_url))
        for game_url in tqdm(urls, desc=f"Downloading {system_name} games data", ncols=120):
            if game_names and not any(name in game_url for name in game_names):
                continue
            game = get_game(game_url)
            download_guides(game)
            with (game_data_path / f"{game.slug}.json").open(mode="wt", encoding="utf-8") as f:
                json.dump(asdict(game), f, cls=DateTimeEncoder)


def get_games_urls(url) -> set[str]:
    text = cached_fetch_text(url, params={"page": 0})
    q = PyQuery(text).make_links_absolute(base_url=f"{url}/")
    search = "table.results tbody tr td.rtitle a:not(.cancel)"
    items = q(search)
    item_urls = {t.attrib["href"] for t in items}
    last_page = max(int(t.attrib.get("value", -1)) for t in q("select#pagejump option"))
    for page in range(1, last_page + 1):
        text = cached_fetch_text(url, params={"page": page})
        q = PyQuery(text).make_links_absolute(base_url=f"{url}/")
        items = q(search)
        item_urls.update(t.attrib["href"] for t in items)
    return item_urls


def get_game(url: str) -> Game:
    slug = Path(urlparse(url).path).name
    text = cached_fetch_text(url)
    q = PyQuery(text).make_links_absolute(base_url=f"{url}/")
    name = q("div.header_right h1.page-title").text()
    description = q("div.content div.game_desc").text()
    game = Game(url=url, slug=slug, name=name, description=description)
    _get_game_fields(q, game)
    game.releases = _get_game_releases(f"{url}/data")
    game.guides = _get_game_guides(f"{url}/faqs")
    return game


def _get_game_fields(q: PyQuery, game: Game) -> None:
    data: dict[str, list[tuple[str, str]]] = {}
    aka_value: list[str] = []
    for item in map(PyQuery, q("div.pod_gameinfo ol li div.content")):
        key = item("b").text().removesuffix(":")
        if key == "Also Known As":
            aka_value = [t.text.removeprefix("• ") for t in item("i")]
        else:
            value: list[tuple[str, str]] = [(t.attrib["href"], t.text) for t in item("a")]  # type: ignore
            if key == "Developer/Publisher":
                data["Developer"] = value
                data["Publisher"] = value
            data[key] = value
    game.platform = [Platform(url, name) for url, name in data["Platform"]]
    game.genre = [Genre(url, name) for url, name in data["Genre"]]
    game.developer = [Developer(url, name) for url, name in data["Developer"]]
    game.publisher = [Publisher(url, name) for url, name in data["Publisher"]]
    game.franchises = [Franchise(url, name) for url, name in data["Franchises"]]
    game.alternative_names = aka_value


def _get_game_releases(url) -> list[Release]:
    text = cached_fetch_text(url)
    q = PyQuery(text).make_links_absolute(base_url=f"{url}/")
    items_q = q("div.gdata div.body table tbody tr")
    _, *items = unflatten(map(PyQuery, items_q), size=2)
    releases = []
    for first_row, second_row in items:
        release = _get_game_release_detail(first_row("td"), second_row("td"))
        releases.append(release)
    return releases


def _get_game_release_detail(first_row: PyQuery, second_row: PyQuery) -> Release:
    _, title_tag = map(PyQuery, first_row)
    (region_tag, publisher_tag, product_id_tag, _, release_date_tag, _) = map(PyQuery, second_row)
    title = title_tag.text()
    region = region_tag.text()
    publisher_q = publisher_tag("a")
    publisher_url = publisher_q.attr("href")
    publisher_name = publisher_q.text()
    product_id = product_id_tag.text()
    release_date_str = release_date_tag.text()
    publisher = Publisher(publisher_url, publisher_name)
    release_date = _parse_date(release_date_str)
    return Release(
        title=title,
        region=region,
        publisher=publisher,
        product_id=product_id,
        release_date=release_date,
    )


def _parse_date(value: str) -> Union[date, str]:
    with suppress(ValueError):
        return datetime.strptime(value, "%m/%d/%y").date()
    with suppress(ValueError):
        return datetime.strptime(value, "%B %Y").date()
    with suppress(ValueError):
        return datetime.strptime(value, "%Y").date()
    if value == "TBA":
        return value
    raise ValueError(f"unknown date value '{value}'")


def _get_game_guides(url: str) -> list[GuideGame]:
    text = cached_fetch_text(url)
    q = PyQuery(text).make_links_absolute(base_url=f"{url}/")
    items_q = q("div.main_content > div.span8 > div.pod:not(#gamespace_search_module):first > *")
    items_section: dict[Union[str, None], list] = {}
    version_name = None
    for el in items_q:
        if el.tag not in ("a", "h2", "div"):
            continue
        if el.tag == "div" and el.attrib["class"] not in ("head", "body"):
            continue
        if el.tag == "a":
            version_name = el.attrib["name"]
            continue
        if el.tag == "div":
            items_section.setdefault(version_name, []).append(el)
    guides_games = []
    for version_key, version_sections in items_section.items():
        sections_items = unflatten(map(PyQuery, version_sections), size=2)
        sections = []
        for head, body in sections_items:
            guilde_items = map(PyQuery, body("ol li"))
            guides = []
            for guilde_item in guilde_items:
                guides.append(_get_game_guides_item(guilde_item))
            sections.append(GuideSection(name=head.text(), guides=guides))
        guides_games.append(GuideGame(name=version_key, sections=sections))
    return guides_games


def _get_game_guides_item(q: PyQuery) -> Guide:
    platform = q.attr("data-platform")
    guide_link = q("a.bold")
    guide_link_url = guide_link.attr("href")
    guide_link_name = guide_link.text()
    author_link = q("span a.link_color")
    author_link_url = author_link.attr("href")
    author_link_name = author_link.text()
    flairs = [flair.text.strip() for flair in q("span.flair")]
    notes = q("div.meta.float_l.bold.ital").text().strip("*")
    status = q("span.ital").text()
    author = GuideAuthor(url=author_link_url, name=author_link_name)
    return Guide(
        platform=platform,
        url=guide_link_url,
        name=guide_link_name,
        author=author,
        flairs=flairs,
        notes=notes,
        status=status,
    )


def download_guides(game: Game) -> None:
    urls = sorted(
        guide.url
        for game_version in game.guides
        for section in game_version.sections
        for guide in section.guides
    )
    for url in tqdm(urls, desc=f"Downloading {game.name} guides", leave=False, ncols=120):
        _download_guide(url)


def _download_guide(url: str) -> None:
    data = cached_fetch(url)
    key = get_cache_key(url)
    path = guides_path / f"{key}.value"
    if not (key_filename := guides_path / f"{key}.key").exists():
        with key_filename.open(mode="xt", encoding="utf-8") as f:
            f.write(url)
    if not path.exists():
        path.mkdir(parents=True)
    mimetype = mime_database.from_buffer(data[:1000])
    if mimetype == "application/zip":
        logger.info("%s is zipfile", url)
        _save_guide_zip(path, data)
        return
    q = PyQuery(data).make_links_absolute(base_url=f"{url}/")
    if (html := q("div.ffaq.ffaqbody#faqwrap").html(method="html")) is not None:
        logger.info("%s is html with size=%d", url, len(html))
        _save_guide_html(path, html)
        return
    if (img_url := q("div#map_container img#gf_map").attr("src")) is not None:
        logger.info("%s contains image as %s", url, img_url)
        _save_guide_img(path, img_url)
        return
    if text := "".join(t.text for t in q("div.faqtext#faqtext pre")):
        logger.info("%s is text with size=%d", url, len(text))
        _save_guide_text(path, text)
        return
    raise ValueError(f"unknown guide type for {url}")


def _save_guide_zip(path: Path, data: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
        for zinfo in zf.infolist():
            if (filename := path / zinfo.filename).exists():
                logger.info("%s exists", filename)
            else:
                zf.extract(zinfo, path=path)
                logger.info("%s saved", filename)


def _save_guide_html(path: Path, html: str) -> None:
    q = PyQuery(html)
    toc = q("div.ftoc").html(method="html")
    for url in sorted(_get_html_guide_toc_links(html)):
        html_filename = _download_html_guide_html(path, url=url)
        toc = toc.replace(url, html_filename)
    if (filename := path / "guide.html").exists():
        logger.info("%s exists", filename)
    else:
        with filename.open(mode="xt", encoding="utf-8") as f:
            f.write(toc)
        logger.info("%s saved", filename)


def _get_html_guide_toc_links(html: str) -> set[str]:
    html_q = PyQuery(html)
    toc_urls = set()
    for t in html_q("div.ftoc a"):
        toc_url = t.attrib["href"]
        parsed_url = urlparse(toc_url)
        parsed_url = parsed_url._replace(fragment="")
        toc_url = urlunparse(parsed_url)
        toc_urls.add(toc_url)
    return toc_urls


def _download_html_guide_html(path: Path, url: str) -> str:
    data = cached_fetch(url)
    key = get_cache_key(url)
    q = PyQuery(data).make_links_absolute(base_url=f"{url}/")
    body = q("div.ffaq.ffaqbody#faqwrap")
    body("div.ftoc").remove()
    html = body.html(method="html").strip()
    html_q = PyQuery(html)
    for img in html_q("img"):
        img_url = img.attrib["src"]
        img_filename = _download_html_guide_image(path, url=img_url)
        html = html.replace(img_url, img_filename)
    if (filename := path / f"{key}.html").exists():
        logger.info("%s exists", filename)
    else:
        with filename.open(mode="xt", encoding="utf-8") as f:
            f.write(html)
        logger.info("%s saved", filename)
    return filename.name


def _download_html_guide_image(path: Path, url: str) -> str:
    data = cached_fetch(url=url)
    key = get_cache_key(url)
    mimetype = mime_database.from_buffer(data)
    ext = mimetypes.guess_extension(mimetype, strict=True)
    if (img_filename := path / f"{key}{ext}").exists():
        logger.info("%s exists", img_filename)
    else:
        with img_filename.open(mode="xb") as f:
            f.write(data)
        logger.info("%s saved", img_filename)
    return img_filename.name


def _save_guide_img(path: Path, img_url: str) -> None:
    img_data = cached_fetch(url=img_url)
    img_mimetype = mime_database.from_buffer(img_data)
    img_ext = mimetypes.guess_extension(img_mimetype, strict=True)
    if (filename := path / f"guide{img_ext}").exists():
        logger.info("%s exists", filename)
    else:
        with filename.open(mode="xb") as f:
            f.write(img_data)
        logger.info("%s saved", filename)


def _save_guide_text(path: Path, text: str) -> None:
    if (filename := path / "guide.txt").exists():
        logger.info("%s exists", filename)
    else:
        with filename.open(mode="xt", encoding="utf-8") as f:
            f.write(text)
        logger.info("%s saved", filename)


def cached_fetch_text(*args, **kwargs) -> str:
    return cached_fetch(*args, **kwargs).decode(encoding="utf-8")


def cached_fetch(url: str, params: Optional[dict[str, Union[str, int]]] = None) -> bytes:
    if params is None:
        params = {}
    params_str = urlencode(sorted(params.items()))
    cache_key = f"{url}::{params_str}"
    try:
        data = cache_get(cache_key)
    except CacheMiss:
        r = _request(url=url, params=params)
        data = r.content
        cache_write(cache_key, data, metadata={"headers": {k: v for k, v in r.headers.items()}})
    return data


def cache_get(key: str) -> bytes:
    logger.info("cache get key=%s", key)
    cache_key = get_cache_key(key)
    try:
        with (cache_path / f"{cache_key}.value").open(mode="rb") as f:
            value = f.read()
    except FileNotFoundError as e:
        logger.warning("cache miss key=%s", key)
        raise CacheMiss(key) from e
    logger.debug("cache hit key=%s, size=%d", key, len(value))
    return value


def cache_write(key: str, value: bytes, metadata: dict[str, Any]) -> None:
    cache_key = get_cache_key(key)
    with (cache_path / f"{cache_key}.value").open(mode="xb") as f:
        f.write(value)
    with (cache_path / f"{cache_key}.key").open(mode="xt", encoding="utf-8") as f:
        f.write(key)
    with (cache_path / f"{cache_key}.metadata").open(mode="xt", encoding="utf-8") as f:
        json.dump(metadata, f)
    logger.info("cache write key=%s, size=%d", key, len(value))


def get_cache_key(key: str) -> str:
    return hashlib.sha256(key.encode(encoding="utf-8")).hexdigest()


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    stop=stop_after_attempt(5),
    wait=wait_fixed(30),
    before=before_log(logger, logging.DEBUG),
    after=after_log(logger, logging.DEBUG),
)
def _request(url: str, params: Optional[dict[str, Union[str, int]]] = None) -> requests.Response:
    r = session.request(method="GET", url=url, params=params)
    r.raise_for_status()
    return r


def unflatten(seq: Iterable, size: int) -> list:
    return [*zip(*(iter(seq),) * size)]


if __name__ == "__main__":
    cli()
