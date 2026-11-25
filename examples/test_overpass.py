import urllib.parse
import urllib.request

url = "https://overpass-api.de/api/interpreter"

query = """
[out:xml][timeout:90];
way(40.94,29.38,40.97,29.43)["highway"="raceway"];
out body;
>;
out skel qt;
"""

data = urllib.parse.urlencode({"data": query}).encode("utf-8")

req = urllib.request.Request(
    url,
    data=data,
    headers={
        "User-Agent": "RIS502-Simstar-Project/1.0",
        "Accept": "*/*",
    }
)

try:
    with urllib.request.urlopen(req, timeout=120) as response:
        text = response.read().decode("utf-8", errors="ignore")
        print("STATUS:", response.status)
        print(text[:1000])
except Exception as e:
    print("ERROR:", e)
    if hasattr(e, "read"):
        print(e.read().decode("utf-8", errors="ignore")[:2000])