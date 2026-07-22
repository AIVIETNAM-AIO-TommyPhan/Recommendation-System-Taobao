
# The ad hierarchy

customer (advertiser, 213)   ← who pays
  └─ campaign_id (248)       ← a budget/strategy grouping
       └─ adgroup_id (300)   ← the actual ad unit shown to the user
            └─ one item → one cate_id (44), one brand (116, incl. the `0`
                          = unknown sentinel → 115 real), one price


a seller (customer) wants to promote their listing
  → creates a campaign (budget/schedule)
    → creates an ad group (adgroup_id) that points at ONE listing
      → that listing has a category, a brand, a price

## adgroup_id — the leaf. 
One ad group = one item being advertised. This is the thing that actually appears in the slot and the thing the user clicks. In your data it functionally determines cate_id, brand, and price: I verified all 300 map to exactly one value of each, no exceptions. So those four columns are one entity's attributes, not four independent signals.

I checked whether the 300 ad groups look like 300 distinct products. They mostly do, but not cleanly:

- 300 ad groups → only 273 distinct (cate_id, brand, price) triples
- 23 triples are shared by more than one ad group, covering 50 ad groups

Some of those are clearly the same seller running two ads for one listing — e.g. ad groups 773194 and 671442 are both category 4281, price 98, customer 164284. Same advertiser, same category, same price. Almost certainly one product, two ad groups. 11 of the 23 shared triples are like this: all the ad groups sit under one customer.

The other 12 are different sellers with coincidentally identical attributes — 735933 and 604083 share category 1665 and price 388 but belong to customers 246083 and 7915. Could be the same product from two shops, could be two different products that happen to cost the same. You cannot tell from this data.

**Caveat on the collision count.** 15 of the 23 shared triples have `brand = 0`, which is the *unknown* sentinel, not a brand. Those triples are only matching on (category, price) in practice, so the collisions are partly an artefact of missing brand data rather than evidence of genuinely duplicated products. The 273-distinct figure is a lower bound on how many real products there are.

## campaign_id — the parent. 
An advertiser groups related ad groups into a campaign to share budget, bid strategy, and date range. Also clean-nested here: every campaign maps to exactly one customer.

In this sample the level barely does any grouping, though: the top-300 filter left 248 campaigns for 300 ad groups, and **217 of them contain exactly one ad group**. So `campaign_id` is close to a copy of `adgroup_id` — encoding both is near-duplicate work. Same for `customer` (213 values). `EDA_REPORT.md` §6 recommends dropping both and keeping `cate_id`/`brand` as the backoff level instead.

All the hierarchy claims above were re-verified against the current `dataset/` files by `eda_scripts/verify_docs.py`.

## pid — the ad slot

Nothing to do with the ad; it identifies where on the page the impression happened ("scenario" / resource position). Two values, and they behave differently:

| pid | impressions (train) | CTR (train) | impressions (test) | CTR (test) |
|---|---|---|---|---|
| `430539_1007` | 94,687 | 21.25% | 22,253 | 20.36% |
| `430548_1007` | 145,313 | 19.49% | 37,747 | 18.61% |

The values are opaque — Taobao never published a mapping to actual page positions, so you can't say which is "top of search results". But the CTR gap is the classic position bias signal: identical ads get clicked at different rates purely from placement. That's why it's a useful feature and why the baseline includes it.

The ordering is stable across the split — `430539_1007` leads in both windows, and the ~1.7pp gap survives the overall CTR drop from 20.19% to 19.26%. So the effect is a property of the slot, not an artefact of one time window. Note the size of the effect, though: `pid` is one of the *weaker* features in the dataset (weighted CTR SD 0.0086, versus 0.0532 for `adgroup_id`) — see `EDA_REPORT.md` §5.