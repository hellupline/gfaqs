#!/usr/bin/env python3

import json
from operator import itemgetter
from pprint import pformat

from pathlib import Path
from typing import Any

from nicegui import ui


@ui.page("/")
async def root():
    with ui.table(columns=games_index_columns, rows=games_index_rows).classes("w-full") as table:
        table.add_slot(
            "body-cell-link",
            """
            <q-td :props="props">
                <a :href="`/game/${props.row.slug}`" class="nicegui-link">
                    {{ props.row.name }}
                </a>
            </q-td>
            """,
        )
        with table.add_slot("top-right"):
            with ui.input(placeholder="search ...").props("type=search") as search:
                with search.add_slot("append"):
                    ui.icon("search")
                search.bind_value(table, "filter")


@ui.page("/game/{slug}")
async def game_detail(slug: str):
    game = games_db[slug]
    for game_version in game["guides"]:
        if game_version["name"] is not None:
            ui.label(f"game_version name: {repr(game_version['name'])}")
        for section in game_version["sections"]:
            ui.label(section["name"])
            rows = [
                {"slug": game["slug"], "key": guide["key"], "name": guide["name"]}
                for guide in section["guides"]
            ]
            with ui.table(columns=games_guides_columns, rows=rows).classes("w-full") as table:
                table.add_slot("header")
                table.add_slot(
                    "body-cell-name",
                    """
                    <q-td :props="props">
                        <a :href="`/game/${props.row.slug}/guide/${props.row.key}`" class="nicegui-link">
                            {{ props.row.name }}
                        </a>
                    </q-td>
                    """,
                )


def get_items() -> list[dict[str, Any]]:
    return [*map(_open_json, game_data_path.rglob("*.json"))]


def _open_json(path: Path) -> dict[str, Any]:
    with path.open(mode="r") as f:
        return json.load(f)


game_data_path = Path("data/games")
games_db = {item["slug"]: item for item in get_items()}
games_index_columns = [
    {
        "name": "link",
        "label": "Link",
        "field": "link",
        "required": True,
        "sortable": True,
        "align": "left",
    },
    {
        "name": "name",
        "label": "Name",
        "field": "name",
        "required": True,
        "sortable": True,
        "align": "left",
    },
    {
        "name": "platform",
        "label": "Platform",
        "field": "platform",
        "required": True,
        "sortable": True,
        "align": "left",
    },
    {
        "name": "genre",
        "label": "Genre",
        "field": "genre",
        "required": True,
        "sortable": True,
        "align": "left",
    },
    {
        "name": "developer",
        "label": "Developer",
        "field": "developer",
        "required": True,
        "sortable": True,
        "align": "left",
    },
    {
        "name": "publisher",
        "label": "Publisher",
        "field": "publisher",
        "required": True,
        "sortable": True,
        "align": "left",
    },
    {
        "name": "franchises",
        "label": "Franchises",
        "field": "franchises",
        "required": True,
        "sortable": True,
        "align": "left",
    },
    # {
    #     "name": "alternative_names",
    #     "label": "Alternative Names",
    #     "field": "alternative_names",
    #     "required": True,
    #     "sortable": True,
    #     "align": "left",
    # },
]
games_guides_columns = [
    {
        "name": "name",
        "label": "Name",
        "field": "name",
        "required": True,
        "sortable": True,
        "align": "left",
    },
]
games_index_rows = [
    {
        "slug": item["slug"],
        "name": item["name"],
        "platform": ", ".join(map(itemgetter("name"), item["platform"])),
        "genre": ", ".join(map(itemgetter("name"), item["genre"])),
        "developer": ", ".join(map(itemgetter("name"), item["developer"])),
        "publisher": ", ".join(map(itemgetter("name"), item["publisher"])),
        "franchises": ", ".join(map(itemgetter("name"), item["franchises"])),
        "alternative_names": ", ".join(item["alternative_names"]),
    }
    for item in games_db.values()
]


ui.run()
