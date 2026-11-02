import json
r = json.load(open("runs/p0_audit.json"))
for rid, x in r["runs"].items():
    print("==", x["name"], "rows", x["rows"], "| limit@14:50 %.3f limit@close %.3f limit@next_open %.3f" % (x["share_limit_at_1450"], x["share_limit_at_close"], x["share_limit_next_open"]))
    for k in ("segments_all", "segments_tradable"):
        print("  ", k)
        for s, v in x[k].items():
            print("     %-16s IC %.4f  RankIC %.4f  ICIR %.2f  rows %d" % (s, v["ic"], v["rank_ic"], v["icir"], v["rows"]))
    print("   winsorized label  IC %.4f RankIC %.4f" % (x["label_winsorized"]["ic"], x["label_winsorized"]["rank_ic"]))
    print("   drop top1%% |label| IC %.4f RankIC %.4f" % (x["label_without_top1pct_abs"]["ic"], x["label_without_top1pct_abs"]["rank_ic"]))
    for s, d in x["deciles_bp"].items():
        print("   deciles(bp, tradable) %-15s" % s, d)
