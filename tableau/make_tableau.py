"""Build tableau/nasipadang_2030.twbx from nasipadang_ranking.csv and the dataset's country labels.

Run from the project folder after the notebook: uv run --with tableauhyperapi python tableau/make_tableau.py

Tableau Public refuses to open a workbook whose data source is not an extract, so the package carries
a Hyper extract of the CSV as well as the CSV itself. tableauhyperapi is only needed here, hence --with.
"""
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr as q

import pandas as pd
from tableauhyperapi import Connection, CreateMode, HyperProcess, Inserter, NOT_NULLABLE, NULLABLE, SqlType, TableDefinition, TableName, Telemetry

OUT = Path("tableau")
rank = pd.read_csv("nasipadang_ranking.csv")
meta = pd.read_excel("data/climate_change.xls", sheet_name="Country").set_index("Country code")
# The dataset uses old World Bank codes for these, which Tableau's map cannot place
ISO3_FOR_LEGACY_CODE = {"ROM": "ROU", "ZAR": "COD", "WBG": "PSE", "TMP": "TLS"}
data = pd.DataFrame({
    "Country code": rank["country_code"].replace(ISO3_FOR_LEGACY_CODE), "Country": rank["name"],
    "Region": rank["country_code"].map(meta["Region"]), "Income group": rank["country_code"].map(meta["Income group"]),
    "Market type": rank["market_type"], "Type probability": rank["type_probability"],
    "Rank 2030": rank["rank"], "Rank today": rank["rank_today"], "Rank change": rank["rank_today"] - rank["rank"],
    "Score": rank["score"], "Top 10 share": rank["top10_share"], "Demand": rank["Demand"],
    "Rice and beef supply": rank["Rice and beef supply"], "Climate and policy": rank["Climate and policy"],
    "Kitchen and operations": rank["Kitchen and operations"],
})
csv_path = OUT / "nasipadang_2030.csv"
data.to_csv(csv_path, index=False, encoding="utf-8")

HYPER_NAME = "nasipadang_2030.hyper"
FED = "federated.nasipadang2030"
TXT = "textscan.nasipadang2030"
TYPES = {"Country code": "string", "Country": "string", "Region": "string", "Income group": "string", "Market type": "string",
         "Type probability": "real", "Rank 2030": "integer", "Rank today": "integer", "Rank change": "integer", "Score": "real",
         "Top 10 share": "real", "Demand": "real", "Rice and beef supply": "real", "Climate and policy": "real", "Kitchen and operations": "real"}
# The same colours as the notebook's market type charts, matched by each type's position in the notebook's order
COLOURS = dict(zip(rank.sort_values("cluster")["market_type"].drop_duplicates(), ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))


def write_extract(path):
    """Write the data to a Hyper file with the Extract.Extract table Tableau expects of an extract."""
    sql_types = {"string": SqlType.text(), "real": SqlType.double(), "integer": SqlType.big_int()}
    columns = [TableDefinition.Column(n, sql_types[t], NULLABLE if data[n].isna().any() else NOT_NULLABLE) for n, t in TYPES.items()]
    table = TableDefinition(TableName("Extract", "Extract"), columns)
    convert = {"string": str, "real": float, "integer": int}
    rows = [[None if pd.isna(v) else convert[t](v) for v, t in zip(row, TYPES.values())] for row in data.itertuples(index=False)]
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters={"log_config": ""}) as hyper:
        with Connection(hyper.endpoint, path, CreateMode.CREATE_AND_REPLACE) as connection:
            connection.catalog.create_schema("Extract")
            connection.catalog.create_table(table)
            with Inserter(connection, table) as inserter:
                inserter.add_rows(rows)
                inserter.execute()


def dim(name):
    return f"[{FED}].[none:{name}:nk]"


def measure(name):
    return f"[{FED}].[sum:{name}:qk]"


def role_of(name):
    if TYPES[name] == "string":
        return "dimension", "nominal"
    return ("dimension", "ordinal") if name.startswith("Rank") else ("measure", "quantitative")


def column(name):
    role, kind = role_of(name)
    return f"<column datatype={q(TYPES[name])} name={q('[' + name + ']')} role={q(role)} type={q(kind)} />"


def datasource_columns():
    out = []
    for name in TYPES:
        role, kind = role_of(name)
        semantic = " semantic-role='[Country].[ISO3166_3]'" if name == "Country code" else ""
        out.append(f"<column datatype={q(TYPES[name])} name={q('[' + name + ']')} role={q(role)}{semantic} type={q(kind)} />")
    return "\n".join(out)


relation_columns = "\n".join(f"<column datatype={q(t)} name={q(n)} ordinal='{i}' />" for i, (n, t) in enumerate(TYPES.items()))
palette = "".join(f"<map to='{v}'><bucket>&quot;{escape(k)}&quot;</bucket></map>" for k, v in COLOURS.items())
COLOUR_STYLE = f"<style-rule element='mark'><encoding attr='color' field='[none:Market type:nk]' type='palette'>{palette}</encoding></style-rule>"
extract_cols = "".join(f"<map key={q('[' + n + ']')} value={q('[Extract].[' + n + ']')} />" for n in TYPES)
EXTRACT = f"""<extract count='-1' enabled='true' units='records'>
<connection access_mode='readonly' author-locale='en_GB' class='hyper' dbname='Data/Extracts/{HYPER_NAME}' default-settings='yes' schema='Extract' sslmode='' tablename='Extract' update-time='{datetime.now():%m/%d/%Y %I:%M:%S %p}' username='tableau_internal_user'>
<relation name='Extract' table='[Extract].[Extract]' type='table' />
<cols>{extract_cols}</cols>
</connection>
</extract>"""
DATASOURCE = f"""<datasource caption='nasipadang_2030' inline='true' name={q(FED)} version='18.1'>
<connection class='federated'>
<named-connections><named-connection caption='nasipadang_2030' name={q(TXT)}>
<connection class='textscan' directory='Data' filename='nasipadang_2030.csv' password='' server='' />
</named-connection></named-connections>
<relation connection={q(TXT)} name='nasipadang_2030.csv' table='[nasipadang_2030#csv]' type='table'>
<columns character-set='UTF-8' header='yes' locale='en_GB' separator=','>{relation_columns}</columns>
</relation>
</connection>
{datasource_columns()}
<column-instance column='[Market type]' derivation='None' name='[none:Market type:nk]' pivot='key' type='nominal' />
{EXTRACT}
<layout dim-ordering='alphabetic' dim-percentage='0.5' measure-ordering='alphabetic' measure-percentage='0.4' show-structure='true' />
<style>{COLOUR_STYLE}</style>
</datasource>"""


# Shelf kind: (derivation, name prefix, column-instance type, name suffix)
KINDS = {"nk": ("None", "none", "nominal", "nk"), "ok": ("None", "none", "ordinal", "ok"), "qk": ("Sum", "sum", "quantitative", "qk"),
         "none-qk": ("None", "none", "quantitative", "qk"), "attr-nk": ("Attribute", "attr", "nominal", "nk"), "attr-ok": ("Attribute", "attr", "ordinal", "ok")}


def instance(name, kind):
    derivation, prefix, kind_type, suffix = KINDS[kind]
    return f"<column-instance column={q('[' + name + ']')} derivation={q(derivation)} name={q(f'[{prefix}:{name}:{suffix}]')} pivot='key' type={q(kind_type)} />"


def sheet(name, title, deps, rows, cols, mark, encodings, extra_view="", mapsources=""):
    dependencies = "".join(column(n) for n in dict(deps)) + "".join(instance(n, k) for n, k in deps)
    return f"""<worksheet name={q(name)}>
<layout-options><title><formatted-text><run bold='true' fontsize='13'>{escape(title)}</run></formatted-text></title></layout-options>
<table>
<view>
<datasources><datasource caption='nasipadang_2030' name={q(FED)} /></datasources>
{mapsources}
<datasource-dependencies datasource={q(FED)}>{dependencies}</datasource-dependencies>
{extra_view}
<aggregation value='true' />
</view>
<style />
<panes><pane selection-relaxation-option='selection-relaxation-allow'>
<view><breakdown value='auto' /></view>
<mark class={q(mark)} />
<encodings>{encodings}</encodings>
</pane></panes>
<rows>{escape(rows)}</rows>
<cols>{escape(cols)}</cols>
</table>
</worksheet>"""


RANK_FILTER = f"[{FED}].[none:Rank 2030:qk]"
TIPS = f"<tooltip column={q(f'[{FED}].[attr:Country:nk]')} /><tooltip column={q(f'[{FED}].[attr:Rank 2030:ok]')} />"
TIP_DEPS = [("Country", "attr-nk"), ("Rank 2030", "attr-ok")]
GEO = ("<column aggregation='Avg' datatype='real' name='[Latitude (generated)]' role='measure' semantic-role='[Geographical].[Latitude]' type='quantitative' />"
       "<column aggregation='Avg' datatype='real' name='[Longitude (generated)]' role='measure' semantic-role='[Geographical].[Longitude]' type='quantitative' />"
       "<column aggregation='Collect' datatype='spatial' name='[Geometry (generated)]' role='measure' type='quantitative' />")
MAP, TOP, ROUTES, BOARD = "Market types map", "Top 20 markets", "Operations and climate scores", "Nasi padang 2030"
map_sheet = sheet(MAP, "Market types by country, 2030",
                  [("Country code", "nk"), ("Market type", "nk"), *TIP_DEPS],
                  f"[{FED}].[Latitude (generated)]", f"[{FED}].[Longitude (generated)]", "Multipolygon",
                  f"<lod column={q(dim('Country code'))} /><color column={q(dim('Market type'))} />{TIPS}<geometry column={q(f'[{FED}].[Geometry (generated)]')} />",
                  mapsources="<mapsources><mapsource name='Tableau' /></mapsources>",
                  ).replace("</datasource-dependencies>", GEO + "</datasource-dependencies>", 1)
top_sheet = sheet(TOP, "Top 20 countries by 2030 score",
                  [("Country", "nk"), ("Market type", "nk"), ("Rank 2030", "none-qk"), ("Score", "qk")],
                  dim("Country"), measure("Score"), "Bar", f"<color column={q(dim('Market type'))} />",
                  extra_view=(f"<filter class='quantitative' column={q(RANK_FILTER)} included-values='in-range'><min>1</min><max>20</max></filter>"
                              f"<sort class='computed' column={q(dim('Country'))} direction='DESC' using={q(measure('Score'))} />"
                              f"<slices><column>{RANK_FILTER}</column></slices>"))
routes_sheet = sheet(ROUTES, "Kitchen and operations score against climate and policy score, by country",
                     [("Country", "nk"), ("Market type", "nk"), *TIP_DEPS, ("Climate and policy", "qk"), ("Kitchen and operations", "qk")],
                     measure("Kitchen and operations"), measure("Climate and policy"), "Circle",
                     f"<lod column={q(dim('Country'))} /><color column={q(dim('Market type'))} />{TIPS}")


def zone(zone_id, name, x, y, w, h):
    return f"<zone h='{h}' id='{zone_id}' name={q(name)} w='{w}' x='{x}' y='{y}' />"


dashboard = f"""<dashboard name={q(BOARD)}>
<style />
<size maxheight='900' maxwidth='1400' minheight='900' minwidth='1400' />
<zones>
<zone h='100000' id='1' type-v2='layout-basic' w='100000' x='0' y='0'>
{zone(2, MAP, 0, 0, 84000, 52000)}
<zone h='52000' id='5' name={q(MAP)} param={q(dim('Market type'))} type-v2='color' w='16000' x='84000' y='0' />
{zone(3, TOP, 0, 52000, 50000, 48000)}
{zone(4, ROUTES, 50000, 52000, 50000, 48000)}
</zone>
</zones>
</dashboard>"""
CARDS = ("<cards><edge name='left'><strip size='160'><card type='pages' /><card type='filters' /><card type='marks' /></strip></edge>"
         "<edge name='top'><strip size='2147483647'><card type='columns' /></strip><strip size='2147483647'><card type='rows' /></strip>"
         "<strip size='31'><card type='title' /></strip></edge></cards>")
worksheet_windows = "".join(f"<window class='worksheet' name={q(n)}>{CARDS}</window>\n" for n in [MAP, TOP, ROUTES])
viewpoints = "".join(f"<viewpoint name={q(n)}><zoom type='entire-view' /></viewpoint>" for n in [MAP, TOP, ROUTES])
twb = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook original-version='18.1' source-build='2026.3.0 (20263.26.1022.1645)' source-platform='win' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
<preferences><preference name='ui.encoding.shelf.height' value='24' /><preference name='ui.shelf.height' value='26' /></preferences>
<datasources>{DATASOURCE}</datasources>
<worksheets>{map_sheet}{top_sheet}{routes_sheet}</worksheets>
<dashboards>{dashboard}</dashboards>
<windows source-height='30'>
{worksheet_windows}<window class='dashboard' maximized='true' name={q(BOARD)}><viewpoints>{viewpoints}</viewpoints><active id='-1' /></window>
</windows>
</workbook>"""
with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(OUT / "nasipadang_2030.twbx", "w", zipfile.ZIP_DEFLATED) as z:
    hyper_path = Path(tmp) / HYPER_NAME
    write_extract(hyper_path)
    z.writestr("nasipadang_2030.twb", twb)
    z.write(csv_path, "Data/nasipadang_2030.csv")
    z.write(hyper_path, f"Data/Extracts/{HYPER_NAME}")
print(f"Built {len(data)} countries:", data["Market type"].value_counts().to_dict())
