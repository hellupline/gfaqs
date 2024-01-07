#!/usr/bin/env python3

import hashlib
import json
import logging
import mimetypes
import random
import zipfile

from collections.abc import AsyncGenerator
from collections.abc import Generator
from collections.abc import Iterable
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from datetime import datetime
from enum import Enum
from functools import cached_property
from io import BytesIO
from operator import attrgetter
from pathlib import Path
from time import sleep
from typing import Optional
from typing import Protocol
from typing import Self
from typing import Union
from urllib.parse import urlparse

import click
import magic
import uvicorn

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import DictLoader
from jinja2 import Environment
from pyquery import PyQuery
from requests import HTTPError
from requests import Response
from requests.status_codes import codes
from requests_cache import NEVER_EXPIRE
from requests_cache import CachedResponse
from requests_cache import CachedSession
from tenacity import RetryCallState
from tenacity import after_log
from tenacity import before_log
from tenacity import retry
from tenacity import retry_base
from tenacity import stop_after_attempt
from tenacity import wait_fixed
from tqdm import tqdm


Params_T = Optional[dict[str, Union[str, int]]]

DEFAULT_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/112.0.0.0 Safari/537.36"
    ),
    "accept": "text/html",
    "accept-language": "en-US,en;q=0.8",
}


storage_path = Path("data/storage")
storage_path.mkdir(parents=True, exist_ok=True)
games_path = Path("data/games")
games_path.mkdir(parents=True, exist_ok=True)
logs_path = Path("logs")
logs_path.mkdir(parents=True, exist_ok=True)

log_format = "[%(asctime)s:%(levelname)s] %(message)s"
log_datefmt = "%Y-%m-%dT%H:%M:%S%z"
log_filename = logs_path / "{:%Y-%m-%dT%H:%M:%S}.log".format(datetime.now())
logging.basicConfig(
    format=log_format,
    datefmt=log_datefmt,
    filename=log_filename,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mime_database = magic.Magic(mime=True)


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    del app
    global CONTENT_INDEX_HASH, GAMES_LIST, GAMES_INDEX_SLUG
    CONTENT_INDEX_HASH = {key: (kind, items) for key, kind, items in load_content_list()}
    GAMES_LIST = load_games_list()
    GAMES_INDEX_SLUG = {game.slug: game for game in GAMES_LIST}
    yield


app = FastAPI(lifespan=app_lifespan)
app.mount(
    "/data/guides/",
    StaticFiles(directory=storage_path),
    name="data_storage",
)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


class WithName(Protocol):
    name: str


class GuideType(Enum):
    HTML = "html"
    TXT = "txt"
    IMG = "img"
    ZIP = "zip"


class GameSystem(Enum):
    ds = "ds"
    gameboy = "gameboy"
    gba = "gba"
    gbc = "gbc"
    n64 = "n64"
    snes = "snes"


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
class Guide:
    platform: str
    url: str = field(repr=False)
    name: str
    author: GuideAuthor
    flairs: list[str] = field(repr=False)
    notes: str = field(repr=False)
    status: str = field(repr=False)


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
    alternative_names: list[str] = field(default_factory=list)
    platform: list[Platform] = field(default_factory=list)
    genre: list[Genre] = field(default_factory=list)
    developer: list[Developer] = field(default_factory=list)
    publisher: list[Publisher] = field(default_factory=list)
    franchises: list[Franchise] = field(default_factory=list)
    releases: list[Release] = field(default_factory=list)
    guides: list[GuideGame] = field(default_factory=list)

    @classmethod
    def from_file(cls: type[Self], path: Path) -> Self:
        with path.open(mode="rt", encoding="utf-8") as f:
            content = json.load(f)
        return cls(
            url=content["url"],
            slug=content["slug"],
            name=content["name"],
            description=content["description"],
            alternative_names=content["alternative_names"],
            platform=[
                Platform(
                    url=platform["url"],
                    name=platform["name"],
                )
                for platform in content["platform"]
            ],
            genre=[
                Genre(
                    url=genre["url"],
                    name=genre["name"],
                )
                for genre in content["genre"]
            ],
            developer=[
                Developer(
                    url=developer["url"],
                    name=developer["name"],
                )
                for developer in content["developer"]
            ],
            publisher=[
                Publisher(
                    url=publisher["url"],
                    name=publisher["name"],
                )
                for publisher in content["publisher"]
            ],
            franchises=[
                Franchise(
                    url=franchise["url"],
                    name=franchise["name"],
                )
                for franchise in content["franchises"]
            ],
            releases=[
                Release(
                    title=release["title"],
                    region=release["region"],
                    publisher=Publisher(
                        url=release["publisher"]["url"],
                        name=release["publisher"]["name"],
                    ),
                    product_id=release["product_id"],
                    release_date=_try_date(release["release_date"]),
                )
                for release in content["releases"]
            ],
            guides=[
                GuideGame(
                    name=guide_game["name"],
                    sections=[
                        GuideSection(
                            name=guide_section["name"],
                            guides=[
                                Guide(
                                    platform=guide["platform"],
                                    url=guide["url"],
                                    name=guide["name"],
                                    author=GuideAuthor(
                                        url=guide["author"]["url"],
                                        name=guide["author"]["name"],
                                    ),
                                    flairs=guide["flairs"],
                                    notes=guide["notes"],
                                    status=guide["status"],
                                )
                                for guide in guide_section["guides"]
                            ],
                        )
                        for guide_section in guide_game["sections"]
                    ],
                )
                for guide_game in content["guides"]
            ],
        )

    @cached_property
    def first_release_date(self) -> date:
        if not self.releases:
            return date.max
        return min(map(_release_date, self.releases))


@click.group(invoke_without_command=False)
def cli() -> None:
    ...


@cli.command("webapp")
@click.option("--host", "host", type=str, default="0.0.0.0")
@click.option("--port", "port", type=int, default=8000)
def cli_webapp(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)


@cli.command("download")
@click.option(
    "--system-name",
    "system_names",
    type=click.Choice([e.value for e in GameSystem]),
    multiple=True,
    required=True,
)
@click.option(
    "--game-name",
    "game_names",
    type=click.STRING,
    multiple=True,
    required=True,
)
def cli_download(system_names: list[str], game_names: list[str]) -> None:
    with CachedSession(
        "requests_cache",
        expire_after=NEVER_EXPIRE,
        allowable_codes=[codes.ok, codes.not_found],
        allowable_methods=["GET"],
    ) as session:
        session.headers.update(DEFAULT_HEADERS)
        # _request(
        #     session,
        #     url="https://gamefaqs.gamespot.com/switch/281230-pokemon-mystery-dungeon-rescue-team-dx/faqs/79236/how-to-recruit-pokemon",
        # )
        # _download_guide_item(
        #     session,
        #     url="https://gamefaqs.gamespot.com/ds/661226-pokemon-black-version-2/faqs/64482",
        # )
        # _download_guide_item(
        #     session,
        #     url="https://gamefaqs.gamespot.com/ds/920760-metroid-prime-hunters/faqs/78897",
        # )
        # _download_guide_item(
        #     session,
        #     url="https://gamefaqs.gamespot.com/switch/281230-pokemon-mystery-dungeon-rescue-team-dx/faqs/79236",
        # )
        for game_system in map(GameSystem, system_names):
            url = f"https://gamefaqs.gamespot.com/{game_system.value}/category/999-all"
            for url in tqdm(
                sorted(get_games_urls(session, url)),
                desc=f"Downloading {game_system.value} guides",
                leave=True,
                ncols=120,
            ):
                if game_names and not any(name in url for name in game_names):
                    continue
                game = get_game(session, url)
                download_guides(session, game)
                with (games_path / f"{game.slug}.json").open(mode="wt", encoding="utf-8") as f:
                    json.dump(asdict(game), f, cls=DateTimeEncoder)


def get_games_urls(session: CachedSession, url: str) -> set[str]:
    r = _request(session, url=url, params={"page": 0})
    q = PyQuery(r.text).make_links_absolute(base_url=f"{r.url}/")
    search = "table.results tbody tr td.rtitle a:not(.cancel)"
    urls = {t.attrib["href"] for t in q(search)}
    last_page = max(int(t.attrib.get("value", -1)) for t in q("select#pagejump option"))
    for page in range(1, last_page + 1):
        r = _request(session, url=url, params={"page": page})
        q = PyQuery(r.text).make_links_absolute(base_url=f"{r.url}/")
        urls.update({t.attrib["href"] for t in q(search)})
    return urls


def get_game(session: CachedSession, url: str) -> Game:
    r = _request(session, url=url)
    q = PyQuery(r.text).make_links_absolute(base_url=f"{r.url}/")
    slug: str = Path(urlparse(url).path).name
    name: str = q("div.header_right h1.page-title").text()  # type: ignore
    description: str = q("div.content div.game_desc").text()  # type: ignore
    game = Game(url=url, slug=slug, name=name, description=description)
    _get_game_fields(q, game)
    game.releases = _get_game_releases(session, url=f"{url}/data")
    game.guides = _get_game_guides(session, f"{url}/faqs")
    return game


def _get_game_fields(q: PyQuery, game: Game) -> None:
    data: dict[str, list[tuple[str, str]]]
    data = {
        "Platform": [],
        "Genre": [],
        "Developer": [],
        "Publisher": [],
        "Franchises": [],
    }
    aka_value = []
    for item in map(PyQuery, q("div.pod_gameinfo ol li div.content")):
        key: str = item("b").text().removesuffix(":")  # type: ignore
        match key:
            case "Also Known As":
                aka_value = [t.text.removeprefix("• ") for t in item("i")]
            case "Developer/Publisher":
                value = [(t.attrib["href"], t.text) for t in item("a")]  # type: ignore
                data["Developer"] = data["Publisher"] = value
            case _:
                value = [(t.attrib["href"], t.text) for t in item("a")]  # type: ignore
                data[key] = value
    game.alternative_names = aka_value
    game.platform = [Platform(url, name) for url, name in data["Platform"]]
    game.genre = [Genre(url, name) for url, name in data["Genre"]]
    game.developer = [Developer(url, name) for url, name in data["Developer"]]
    game.publisher = [Publisher(url, name) for url, name in data["Publisher"]]
    game.franchises = [Franchise(url, name) for url, name in data["Franchises"]]


def _get_game_releases(session: CachedSession, url: str) -> list[Release]:
    r = _request(session, url=url)
    q = PyQuery(r.text).make_links_absolute(base_url=f"{r.url}/")
    items_q = q("div.gdata div.body table tbody tr")
    _, *items = _unflatten(map(PyQuery, items_q), size=2)
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
    publisher = Publisher(publisher_url, publisher_name)  # type: ignore
    release_date = _parse_date(release_date_str)  # type: ignore
    return Release(
        title=title,  # type: ignore
        region=region,  # type: ignore
        publisher=publisher,  # type: ignore
        product_id=product_id,  # type: ignore
        release_date=release_date,  # type: ignore
    )


def _get_game_guides(session: CachedSession, url: str) -> list[GuideGame]:
    r = _request(session, url=url)
    q = PyQuery(r.text).make_links_absolute(base_url=f"{r.url}/")
    items_q = q("div.main_content > div.span8 > div.pod:not(#gamespace_search_module):first > *")
    items_section: dict[Union[str, None], list] = {}
    version_name = None
    for element in items_q:
        if element.tag not in ("a", "h2", "div"):
            continue
        if element.tag == "div" and element.attrib["class"] not in ("head", "body"):
            continue
        if element.tag == "a":
            version_name = element.attrib["name"]
            continue
        if element.tag == "div":
            items_section.setdefault(version_name, []).append(element)
    guides_games = []
    for version_key, version_sections in items_section.items():
        sections_items = _unflatten(map(PyQuery, version_sections), size=2)
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
    notes = q("div.meta.float_l.bold.ital").text().strip("*")  # type: ignore
    status = q("span.ital").text()
    author = GuideAuthor(url=author_link_url, name=author_link_name)  # type: ignore
    return Guide(
        platform=platform,  # type: ignore
        url=guide_link_url,  # type: ignore
        name=guide_link_name,  # type: ignore
        author=author,
        flairs=flairs,
        notes=notes,
        status=status,  # type: ignore
    )


def download_guides(session: CachedSession, game: Game) -> None:
    urls = sorted(
        guide.url
        for game_version in game.guides
        for section in game_version.sections
        for guide in section.guides
    )
    for url in tqdm(
        urls,
        desc=f"Downloading {game.name} guides",
        leave=False,
        ncols=120,
    ):
        _download_guide_item(session, url=url)


def _download_guide_item(session: CachedSession, url: str) -> None:
    r = _request(session, url=url)
    data = r.content
    mimetype = mime_database.from_buffer(data)
    if mimetype == "application/zip":
        logger.info("%s is zipfile with size=%d", url, len(data))
        _save_guide_zip(url=url, data=data)
        return
    q = PyQuery(r.text).make_links_absolute(base_url=f"{r.url}/")
    if (html := q("div.ffaq.ffaqbody#faqwrap").html(method="html")) is not None:
        logger.info("%s is html with size=%d", url, len(html))
        _save_guide_html(session, key=url, html=html)  # type: ignore
        return
    if (img_url := q("div#map_container img#gf_map").attr("src")) is not None:
        logger.info("%s contains image as %s", url, img_url)
        _save_guide_image(session, key=url, url=img_url)  # type: ignore
        return
    if text := "".join(t.text for t in q("div.faqtext#faqtext pre")):
        logger.info("%s is text with size=%d", url, len(text))
        _save_guide_text(key=url, text=text)
        return
    raise ValueError(f"unknown guide type for {url}")


def _save_guide_zip(url: str, data: bytes) -> None:
    path = _save_data(name="data.zip", key=url, data=data)
    path = path / "content"
    with zipfile.ZipFile(BytesIO(data)) as zf:
        for name in zf.namelist():
            filename = (filename := path / name)
            if filename.exists():
                logger.info("%s exists", filename)
                continue
            filename.parent.mkdir(parents=True, exist_ok=True)
            content_data = zf.read(name)
            filename.write_bytes(content_data)


def _save_guide_html(session: CachedSession, key: str, html: str) -> Path:
    q = PyQuery(html)
    content = q("div.ftoc")
    visited: set[str] = set()
    url_data_map = {
        visited_url: data
        for url in sorted({_clean_url(tag.attrib["href"]) for tag in content("a")})
        for visited_url, data in _download_html_guide_html(session, url=url, visited=visited)
    }
    url_map = {url: _html_url_key(url) for url in url_data_map.keys()}
    for url, data in url_data_map.items():
        data_q = PyQuery(data)
        for tag in data_q("a"):
            if "href" not in tag.attrib:
                continue
            with suppress(KeyError):
                tag.attrib["href"] = _get_new_url(tag.attrib["href"], url_map=url_map)
        _save_data(name="data.html", key=url, data=_as_html_data(data_q))
    for tag in content("a"):
        if "href" not in tag.attrib:
            continue
        with suppress(KeyError):
            tag.attrib["href"] = _get_new_url(tag.attrib["href"], url_map=url_map)
    path = _save_data(name="data.html", key=key, data=_as_html_data(content))
    return path / "data.html"


def _download_html_guide_html(
    session: CachedSession,
    url: str,
    visited: set[str],
) -> Generator[tuple[str, bytes], None, None]:
    if url in visited:
        return
    r = _request(session, url=url)
    q = PyQuery(r.text).make_links_absolute(base_url=r.url)
    content = q("div.ffaq.ffaqbody#faqwrap")
    content("div.ftoc").remove()
    url_path_map = {
        url: _download_html_guide_image(session, url=url)
        for url in sorted(_clean_url(tag.attrib["src"]) for tag in content("img"))
    }
    url_map = {url: str(path.relative_to(storage_path)) for url, path in url_path_map.items()}
    for tag in content("img"):
        tag.attrib["src"] = _get_new_url(tag.attrib["src"], url_map=url_map)
    if content.html(method="html") is None:
        raise Exception(url)
    yield url, _as_html_data(content)
    visited.add(url)
    to_visit_urls = sorted({_clean_url(tag.attrib["href"]) for tag in content("a") if "href" in tag.attrib})
    for to_visit_url in to_visit_urls:
        if to_visit_url in INVALID_URLS:
            continue
        try:
            yield from _download_html_guide_html(session=session, url=to_visit_url, visited=visited)
        except (HTTPError,) as e:
            match e.response:
                case Response(status_code=codes.not_found):
                    continue
            raise


def _download_html_guide_image(session: CachedSession, url: str) -> Path:
    r = _request(session, url=url)
    data = r.content
    mimetype = mime_database.from_buffer(data)
    ext = mimetypes.guess_extension(mimetype, strict=True)
    if ext is None:
        ext = ".bin"
    filename = f"data{ext}"
    path = _save_data(name=filename, key=url, data=data)
    return path / filename


def _save_guide_image(session: CachedSession, key: str, url: str) -> Path:
    r = _request(session, url=url)
    data = r.content
    mimetype = mime_database.from_buffer(data)
    ext = mimetypes.guess_extension(mimetype, strict=True)
    if ext is None:
        ext = ".bin"
    filename = f"data{ext}"
    path = _save_data(name=filename, key=key, data=data)
    return path / filename


def _save_guide_text(key: str, text: str) -> None:
    data = text.encode(encoding="utf-8")
    _save_data(name="data.txt", key=key, data=data)


class retry_if_http_error(retry_base):
    def __call__(self, retry_state: RetryCallState) -> bool:
        if retry_state.outcome is not None:
            match retry_state.outcome.exception():
                case None:
                    return False
                case HTTPError(response=Response(status_code=codes.not_found)):
                    return False
        return True


@retry(
    retry=retry_if_http_error(),
    stop=stop_after_attempt(5),
    wait=wait_fixed(30),
    before=before_log(logger, logging.DEBUG),
    after=after_log(logger, logging.DEBUG),
)
def _request(session: CachedSession, url: str, params: Params_T = None) -> Response:
    logger.info("url request %s params %s", url, repr(params))
    r = session.request(method="GET", url=url, params=params)
    r.raise_for_status()
    if not isinstance(r, (CachedResponse,)):
        logger.info("url not cached %s", url)
        time_to_sleep = random.randint(100, 300) / 100
        sleep(time_to_sleep)
    else:
        logger.info("url cached %s", url)
    return r


# @retry(
#     retry=retry_if_http_error(),
#     stop=stop_after_attempt(5),
#     wait=wait_fixed(30),
#     before=before_log(logger, logging.DEBUG),
#     after=after_log(logger, logging.DEBUG),
# )
# def _stream_request(session: CachedSession, url: str, params: Params_T = None) -> tuple[Response, bytes]:
#     r = session.request(method="GET", url=url, params=params, stream=True)
#     r.raise_for_status()
#     total_size = int(r.headers.get("content-length", 0))
#     block_size = 1024
#     buffer = BytesIO()
#     with tqdm.wrapattr(
#         stream=buffer,
#         method="write",
#         miniters=1,
#         desc=f"Downloading {url}",
#         total=total_size,
#         unit="B",
#         unit_scale=True,
#         leave=False,
#         ncols=120,
#     ) as f:
#         for data in r.iter_content(block_size):
#             f.write(data)
#     buffer.seek(0)
#     if not isinstance(r, (CachedResponse,)):
#         time_to_sleep = random.randint(100, 300) / 100
#         sleep(time_to_sleep)
#     return r, buffer.getvalue()


def _save_data(name: str, key: str, data: bytes) -> Path:
    key_hash = _key_hash(key)
    data_prefix = storage_path / f"{key_hash}.value"
    data_prefix.mkdir(parents=True, exist_ok=True)
    key_filename = storage_path / f"{key_hash}.key"
    key_filename.write_text(key, encoding="utf-8")
    data_filename = data_prefix / name
    # if data_filename.exists():
    #     logger.info("%s exists", data_filename)
    #     return data_prefix
    data_filename.write_bytes(data)
    logger.info("key %s saved as %s ", key, name)
    return data_prefix


def _as_html_data(q: PyQuery) -> bytes:
    content = q.html(method="html")
    if content is None:
        raise ValueError("empty html")
    return content.strip().encode(encoding="utf-8")  # type: ignore


def _html_url_key(url: str) -> str:
    key = _key_hash(url)
    path = storage_path / f"{key}.value" / "data.html"
    return str(path.relative_to(storage_path))


def _clean_url(value: str) -> str:
    return urlparse(value)._replace(fragment="").geturl()


def _get_new_url(url: str, url_map: dict[str, str]) -> str:
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="").geturl()
    fragment = parsed.fragment
    new_url = url_map[normalized]
    return urlparse(new_url)._replace(fragment=fragment).geturl()


def _key_hash(value: str) -> str:
    return hashlib.sha256(value.encode(encoding="utf-8")).hexdigest()


def _unflatten(seq: Iterable, size: int) -> list:
    return [*zip(*(iter(seq),) * size)]


def _parse_date(value: str) -> Union[date, str]:
    with suppress(ValueError):
        return datetime.strptime(value, "%m/%d/%y").date()
    with suppress(ValueError):
        return datetime.strptime(value, "%B %Y").date()
    with suppress(ValueError):
        return datetime.strptime(value, "%Y").date()
    if value == "TBA":
        return value
    if value == "Canceled":
        return value
    raise ValueError(f"unknown date value '{value}'")


@app.get("/", response_class=RedirectResponse)
def root():
    return app.url_path_for("games_list")


@app.get("/games", response_class=HTMLResponse)
def games_list() -> str:
    tpl = template_environment.get_template("games_list.html")
    return tpl.render(games=GAMES_LIST)


@app.get("/games/{slug:str}", response_class=HTMLResponse)
def game_detail(slug: str) -> str:
    game = GAMES_INDEX_SLUG[slug]
    _, game_slug = game.slug.split("-", maxsplit=1)
    tpl = template_environment.get_template("game_detail.html")
    return tpl.render(game=game, game_slug=game_slug)


@app.get("/games/{slug:str}/guide/{key:str}", response_class=HTMLResponse)
def game_guide(slug: str, key: str) -> str:
    game = GAMES_INDEX_SLUG[slug]
    for guide_game in game.guides:
        for guide_section in guide_game.sections:
            for guide in guide_section.guides:
                guide_type, item = CONTENT_INDEX_HASH[key]
                match guide_type:
                    case GuideType.ZIP:
                        return "ZIP"
                    case GuideType.HTML:
                        tpl = template_environment.get_template("game_guide_html_toc.html")
                        text = item.read_text(encoding="utf-8")
                        q = PyQuery(text)
                        for tag in q("a"):
                            if "href" not in tag.attrib:
                                continue
                            item_key = Path(tag.attrib["href"]).parent.stem
                            url = app.url_path_for("game_guide", slug=slug, key=item_key)
                            tag.attrib["href"] = url
                        html = q.html(method="html")
                        return tpl.render(guide=guide, toc=html)
                    case GuideType.IMG:
                        tpl = template_environment.get_template("game_guide_img.html")
                        path = item.relative_to(storage_path)
                        url = app.url_path_for("data_storage", path=str(path))
                        return tpl.render(guide=guide, url=url)
                    case GuideType.TXT:
                        tpl = template_environment.get_template("game_guide_txt.html")
                        text = item.read_text(encoding="utf-8")
                        return tpl.render(guide=guide, text=text)
                    case _:
                        raise ValueError(f"Unknown guide type for {key}")
    raise HTTPException(status_code=404, detail="Guide not found")


def load_content_list() -> Generator[tuple[str, GuideType, Path], None, None]:
    for path in storage_path.glob("*.key"):
        key = path.stem
        guide_path = storage_path / f"{key}.value"
        if (filename := (guide_path / "data.html")).exists():
            yield (key, GuideType.HTML, filename)
        elif (filename := (guide_path / "data.png")).exists():
            yield (key, GuideType.IMG, filename)
        elif (filename := (guide_path / "data.jpg")).exists():
            yield (key, GuideType.IMG, filename)
        elif (filename := (guide_path / "data.gif")).exists():
            yield (key, GuideType.IMG, filename)
        elif (filename := (guide_path / "data.txt")).exists():
            yield (key, GuideType.TXT, filename)
        elif (filename := (guide_path / "data.zip")).exists():
            yield (key, GuideType.ZIP, filename)
        # else:
        #     raise ValueError(f"Unknown guide type for {key}")


def load_games_list() -> list[Game]:
    return sorted(
        (Game.from_file(path) for path in games_path.rglob("*.json")),
        key=attrgetter("first_release_date", "name"),
        reverse=True,
    )


def _try_date(value: str) -> Union[date, str]:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return value


def _release_date(release: Release):
    if isinstance(release.release_date, (date,)):
        return release.release_date
    return date.max


def _get_items_names(items: list[WithName]) -> str:
    return ", ".join(item.name for item in items)


GAMES_LIST_TPL = """{% extends "base.html" %}
{%- block title %}Games list{%- endblock title %}
{%- block content %}
        <h1>Games</h1>
        <table class="table table-stripped table-bordered table-hover">
            <thead>
                <tr>
                    <td>Name</td>
                    <td>Release Date</td>
                </tr>
            </thead>
            <tbody>
                {%- for game in games %}
                <tr>
                    <td><a href="{{ url_path_for("game_detail", slug=game.slug) }}">{{ game.name }}</a></td>
                    <td>{{ game.first_release_date }}</td>
                </tr>
                {%- endfor %}
            </tbody>
        </table>
{%- endblock content %}"""

GAME_DETAIL_TPL = """{% extends "base.html" %}
{%- block title %}{{ game.name }}{%- endblock title %}
{%- block content %}
        <h4>Details</h4>
        <table class="table table-stripped table-bordered">
            <tbody>
                <tr>
                    <td>Name</td>
                    <td>{{ game.name }}</td>
                </tr>
                <tr>
                    <td>Alternative Names</td>
                    <td>{{ game.alternative_names | join(", ") }}</td>
                </tr>
                <tr>
                    <td>Platform</td>
                    <td>{{ get_items_names(game.platform) }}</td>
                </tr>
                <tr>
                    <td>Genre</td>
                    <td>{{ get_items_names(game.genre) }}</td>
                </tr>
                <tr>
                    <td>Developer</td>
                    <td>{{ get_items_names(game.developer) }}</td>
                </tr>
                <tr>
                    <td>Publisher</td>
                    <td>{{ get_items_names(game.publisher) }}</td>
                </tr>
                <tr>
                    <td>Franchises</td>
                    <td>{{ get_items_names(game.franchises) }}</td>
                </tr>
            </tbody>
        </table>
        <h4>Releases</h4>
        <table class="table table-stripped table-bordered">
            <thead>
                <tr>
                    <td>Title</td>
                    <td>Region</td>
                    <td>Publisher</td>
                    <td>Product ID</td>
                    <td>Release Date</td>
                </tr>
            </thead>
            <tbody>
                {%- for release in game.releases %}
                <tr>
                    <td>{{ release.title }}</td>
                    <td>{{ release.region }}</td>
                    <td>{{ release.publisher.name }}</td>
                    <td>{{ release.product_id }}</td>
                    <td>{{ release.release_date }}</td>
                </tr>
                {%- endfor %}
            </tbody>
        </table>
        <h4>Guides</h4>
        {%- if game.guides|length > 1 %}
        <nav>
            <div class="nav nav-tabs" id="nav-tab" role="tablist">
                {%- for guide_game in game.guides %}
                <button
                    class="nav-link{% if guide_game.name == game_slug %} active{% endif %}"
                    id="nav-{{ key_hash(guide_game.name or "") }}-tab"
                    data-bs-toggle="tab"
                    data-bs-target="#nav-{{ key_hash(guide_game.name or "") }}"
                    type="button"
                    role="tab"
                    aria-controls="nav-{{ key_hash(guide_game.name or "") }}"
                    aria-selected="true"
                >
                    {{ guide_game.name }}
                </button>
                {%- endfor %}
            </div>
        </nav>
        <div class="tab-content" id="nav-tabContent">
            {%- for guide_game in game.guides %}
            <div
                class="tab-pane show{% if guide_game.name == game_slug %} active{% endif %}"
                id="nav-{{ key_hash(guide_game.name or "") }}"
                role="tabpanel"
                aria-labelledby="nav-{{ key_hash(guide_game.name or "") }}-tab"
            >
                <table class="table table-stripped table-bordered table-hover">
                    <tbody>
                    {%- for guide_section in guide_game.sections %}
                        <tr>
                            <td colspan="3"><strong>{{ guide_section.name }}</strong></td>
                        </tr>
                        {%- for guide in guide_section.guides %}
                        <tr>
                            <td><a href="{{ url_path_for("game_guide", slug=game.slug, key=key_hash(guide.url)) }}">{{ guide.name }}</a></td>
                            <td>{{ guide.platform }}</td>
                            <td>{{ guide.author.name }}</td>
                        </tr>
                        {%- endfor %}
                    {%- endfor %}
                    </tbody>
                </table>
            </div>
            {%- endfor %}
        </div>
        {%- else %}
        {%- for guide_game in game.guides %}
        <div
            class="tab-pane show{% if guide_game.name == game_slug %} active{% endif %}"
            id="nav-{{ key_hash(guide_game.name or "") }}"
            role="tabpanel"
            aria-labelledby="nav-{{ key_hash(guide_game.name or "") }}-tab"
        >
            <table class="table table-stripped table-bordered table-hover">
                <tbody>
                {%- for guide_section in guide_game.sections %}
                    <tr>
                        <td colspan="3"><strong>{{ guide_section.name }}</strong></td>
                    </tr>
                    {%- for guide in guide_section.guides %}
                    <tr>
                        <td><a href="{{ url_path_for("game_guide", slug=game.slug, key=key_hash(guide.url)) }}">{{ guide.name }}</a></td>
                        <td>{{ guide.platform }}</td>
                        <td>{{ guide.author.name }}</td>
                    </tr>
                    {%- endfor %}
                {%- endfor %}
                </tbody>
            </table>
        </div>
        {%- endfor %}
        {%- endif %}
{%- endblock content %}"""

GAME_GUIDE_HTML_TOC_TPL = """{% extends "base.html" %}
{%- block title %}Guide{%- endblock title %}
{%- block content %}
        {{ toc }}
{%- endblock content %}"""

GAME_GUIDE_HTML_CONTENT_TPL = """{% extends "base.html" %}
{%- block title %}Guide{%- endblock title %}
{%- block content %}
        {{ toc }}
        {{ html }}
{%- endblock content %}"""

GAME_GUIDE_IMG_TPL = """{% extends "base.html" %}
{%- block title %}Guide{%- endblock title %}
{%- block content %}
        <img src="{{ url }}" alt="{{ guide.name }}">
{%- endblock content %}"""

GAME_GUIDE_TXT_TPL = """{% extends "base.html" %}
{%- block title %}{{ guide.name }}{%- endblock title %}
{%- block content %}
        <pre>{{ text }}</pre>
{%- endblock content %}"""

# https://getbootstrap.com/docs/5.3/examples/cheatsheet/
BASE_TPL = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{% block title required %}{% endblock title %}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
        <style type="text/css">
            .ffaq tr, td, th {
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        {%- block content required %}
        {%- endblock content %}
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
    </body>
</html>"""

template_loader = DictLoader(
    {
        "games_list.html": GAMES_LIST_TPL,
        "game_detail.html": GAME_DETAIL_TPL,
        "game_guide_html_toc.html": GAME_GUIDE_HTML_TOC_TPL,
        "game_guide_html_content.html": GAME_GUIDE_HTML_CONTENT_TPL,
        "game_guide_img.html": GAME_GUIDE_IMG_TPL,
        "game_guide_txt.html": GAME_GUIDE_TXT_TPL,
        "base.html": BASE_TPL,
    }
)
template_environment = Environment(loader=template_loader)
template_environment.globals["url_path_for"] = app.url_path_for
template_environment.globals["key_hash"] = _key_hash
template_environment.globals["get_items_names"] = _get_items_names

CONTENT_INDEX_HASH: dict[str, tuple[GuideType, Path]] = {}
GAMES_LIST: list[Game] = []
GAMES_INDEX_SLUG: dict[str, Game] = {}
INVALID_URLS = {
    "https://www.gamefaqs.com/contribute/test_file_v3/51/minigames",
}


if __name__ == "__main__":
    cli()
