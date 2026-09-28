"""Generate a small synthetic dataset with the SAME layout as the challenge data.

Used only to smoke-test the pipeline end to end (never for training a submission):

    python tools/make_synthetic_data.py --out synthetic_dataset --n-train 3000 --n-test 2000
    python run.py all --data-dir synthetic_dataset
"""
import argparse
import os
import random

WORDS = ("acme robotics delta foods bright cafe zen traders kappa motors sharma krishna lotus "
         "global tech sun pharma blue ocean star textiles royal bakery green valley metro steel "
         "apex logistics silver line alpha systems nova digital prime builders urban kitchen "
         "golden gate river stone maple leaf city dental sai balaji ganesh agro vista media "
         "lumiere boulangerie maison atelier provence soleil jardin").split()
SUFFIX = {"US": ["Inc", "LLC", "Corp", "Co", ""], "India": ["Pvt Ltd", "Private Limited", "Ltd", "Enterprises", ""],
          "France": ["SARL", "SAS", "SA", "EURL", ""]}
LONG = {"Inc": "Incorporated", "Corp": "Corporation", "Co": "Company", "Pvt Ltd": "Private Limited",
        "Ltd": "Limited", "SARL": "S.A.R.L."}
STREETS = {"US": ["Market St", "Oak Ave", "Main Street", "Pine Rd", "Elm Blvd"],
           "India": ["MG Road", "Station Rd", "Nehru Nagar", "Gandhi Marg", "Park Street"],
           "France": ["Rue de la Paix", "Bd Voltaire", "Avenue Foch", "Rue Victor Hugo", "Place Bellecour"]}
CITIES = {"US": ["San Jose", "Austin", "Denver", "Boston", "Reno"],
          "India": ["Mumbai", "Pune", "Delhi", "Chennai", "Jaipur"],
          "France": ["Paris", "Lyon", "Nice", "Lille", "Nantes"]}
ABBR = {"Street": "St", "Road": "Rd", "Avenue": "Ave", "Boulevard": "Blvd", "Private": "Pvt",
        "Limited": "Ltd", "Market": "Mkt", "Station": "Stn", "Rue": "R.", "Bd": "Boulevard"}


def entity(rng, country):
    name = " ".join(rng.sample(WORDS, rng.choice([1, 2, 2, 3]))).title()
    suf = rng.choice(SUFFIX[country])
    num = rng.randint(1, 999)
    pc = str(rng.randint(10000, 99999)) if country != "India" else str(rng.randint(110000, 999999))
    addr = f"{num} {rng.choice(STREETS[country])}, {rng.choice(CITIES[country])} {pc}"
    return {"name": f"{name} {suf}".strip(), "addr": addr, "country": country}


def typo(rng, s):
    if len(s) < 5:
        return s
    k = rng.randrange(1, len(s) - 1)
    return s[:k] + s[k + 1:] if rng.random() < 0.5 else s[:k] + s[k + 1] + s[k] + s[k + 2:]


def noisy(rng, e):
    name, addr = e["name"], e["addr"]
    for a, b in LONG.items():
        if rng.random() < 0.5 and name.endswith(a):
            name = name[: -len(a)] + b
    if rng.random() < 0.3:
        name = name.replace(" and ", " & ")
    if rng.random() < 0.25:
        name = typo(rng, name)
    if rng.random() < 0.15:
        name = name.upper()
    for a, b in ABBR.items():
        if a in addr and rng.random() < 0.5:
            addr = addr.replace(a, b)
    if rng.random() < 0.2:
        addr = addr.rsplit(" ", 1)[0]  # drop postcode
    if rng.random() < 0.1:
        addr = "Near " + rng.choice(["SBI ATM", "City Hall", "Bus Stand", "la Mairie"]) + ", " + addr.split(",")[-1]
    return {"name": name, "addr": addr, "country": e["country"]}


def build(rng, n, countries, prefix_start):
    s1, s2, s3, gt = [], [], [], []
    c2 = c3 = 0
    for k in range(n):
        country = rng.choice(countries)
        e = entity(rng, country)
        sid = f"S1-{prefix_start + k:07d}"
        s1.append((sid, e["name"], e["addr"], country))
        matches = []
        r = rng.random()
        n2 = 0 if r < 0.3 else rng.choice([1, 1, 1, 2])
        n3 = 0 if r < 0.3 else rng.choice([0, 1, 1, 2])
        for _ in range(n2):
            v = noisy(rng, e); c2 += 1; cid = f"S2-{prefix_start + c2:07d}"
            s2.append((cid, v["name"], v["addr"], country)); matches.append(cid)
        for _ in range(n3):
            v = noisy(rng, e); c3 += 1; cid = f"S3-{prefix_start + c3:07d}"
            s3.append((cid, v["name"], v["addr"], country)); matches.append(cid)
        # look-alike distractor: same name, other address / similar name same address
        if rng.random() < 0.4:
            d = entity(rng, country)
            d["name"] = e["name"] if rng.random() < 0.5 else e["name"].split()[0] + " " + d["name"]
            c2 += 1; s2.append((f"S2-{prefix_start + c2:07d}", d["name"], d["addr"], country))
        if rng.random() < 0.4:
            d = entity(rng, country)
            c3 += 1; s3.append((f"S3-{prefix_start + c3:07d}", d["name"], e["addr"], country))
        gt.append((sid, ",".join(matches)))
    return s1, s2, s3, gt


def write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_dataset")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cols = ["entity_id", "business_name", "business_address", "country"]
    for split, n, countries, start in (("train", a.n_train, ["US", "India"], 0),
                                       ("test", a.n_test, ["US", "India", "France"], 5_000_000)):
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        s1, s2, s3, gt = build(rng, n, countries, start)
        rng.shuffle(s2); rng.shuffle(s3)
        write(os.path.join(a.out, split, f"{split}_source1.tsv"), cols, s1)
        write(os.path.join(a.out, split, f"{split}_source2.tsv"), cols, s2)
        write(os.path.join(a.out, split, f"{split}_source3.tsv"), cols, s3)
        if split == "train":
            write(os.path.join(a.out, split, "train_ground_truth.tsv"),
                  ["source1_entity_id", "matched_entity_ids"], gt)
    print("synthetic dataset written to", a.out)


if __name__ == "__main__":
    main()
