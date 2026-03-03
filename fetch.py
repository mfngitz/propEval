import requests
import json
import os

tok = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjF0dUkybDZSQjBjWlF2MHM1M28yNSJ9.eyJzdWJzY3JpcHRpb24iOiJ0cmlhbCIsImlzcyI6Imh0dHBzOi8vcHJvcHMtaGVscGVyLnVzLmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzc5NTA2MTM1MjIxNzU0NzU2MiIsImF1ZCI6WyJodHRwczovL3Byb3BzLWRvdC1jYXNoL2FwaSIsImh0dHBzOi8vcHJvcHMtaGVscGVyLnVzLmF1dGgwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3NzE3MTYxNzMsImV4cCI6MTc3NDMwODE3Mywic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCBvZmZsaW5lX2FjY2VzcyIsImF6cCI6ImtJNzZQWWs5QTNnN3lVdHloWTNBaEttcjlvdmlIQXp3In0.RfhJ_j0hjmVqgm3QEhiQ2p3BWFfhnLuJ25153UxVr4cqU7lbY256ipWL34qHj09CI5DewwZ7DIjM4WfdavRnTzA_5NiFqUJ5H1b6hddSWW8mwl_y3h5QYKveG04iLrUkrT1WnQF1jBHLN1dRV1ba790QV0Pj2L9ksebm1DQ7SQflyRWe3nhLZH6Thz2jo8n_Qkio7AFlDYRu_Egjs79jCk9sJE88oeHD_SN91w0a90rcT227m4XpW-a3PlJJlmE7-4D2I57W_JtIK2wP6KK3-Bq_BQ2CRJS1ZZeuOkNaT2dXqVfj3xETih4gIZi9vk6V1RzY10lpZjENNfE_JKYV0g"

url_t = "https://api.props.cash/NBA/prop-trends"
url_p = "https://api.props.cash/nba/projections"
url_a = "https://api.props.cash/nba/alt-lines"
url_s = "https://api.props.cash/nba/schedule"

hdr = {
    "Authorization": f"Bearer {tok}",
    "Accept": "application/json",
    "Origin": "https://props.cash",
    "Referer": "https://props.cash/",
    "User-Agent": "Mozilla/5.0"
}

rm = {
    "q1Assists",
    "q1Rebounds",
    "doubleDouble",
    "blocks",
    "dunks",
    "firstBasket",
    "fgMade",
    "steals",
    "stealsAndBlocks",
    "fg3PtMade",
    "turnovers"
}

os.makedirs("jsons", exist_ok=True)


def f_trn(dat):
    for pl in dat:
        for k in rm:
            pl.pop(k, None)
    return dat


def f_prj(dat):
    for pl in dat:
        if "projections" in pl:
            for k in rm:
                pl["projections"].pop(k, None)
    return dat


def f_alt(dat):
    return [x for x in dat if x.get("prop") not in rm]


def save(name, data):
    with open(f"jsons/{name}", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


r = requests.get(url_t, headers=hdr)
if r.status_code != 200:
    print("trn err:", r.status_code)
    exit()
save("prop_trends.json", f_trn(r.json()))
print("saved jsons/prop_trends.json")


r = requests.get(url_p, headers=hdr)
if r.status_code != 200:
    print("prj err:", r.status_code)
    exit()
save("projections.json", f_prj(r.json()))
print("saved jsons/projections.json")


r = requests.get(url_a, headers=hdr)
if r.status_code != 200:
    print("alt err:", r.status_code)
    exit()
save("alt_lines.json", f_alt(r.json()))
print("saved jsons/alt_lines.json")


r = requests.get(url_s, headers=hdr)
if r.status_code != 200:
    print("sch err:", r.status_code)
    exit()
save("schedule.json", r.json())
print("saved jsons/schedule.json")
