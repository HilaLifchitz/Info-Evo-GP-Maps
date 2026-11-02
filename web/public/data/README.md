# Precomputed results

Python should write JSON here (for example `eta_maint.json`) after equilibrium runs. The site will `fetch` those files; nothing is computed in the browser.

Schema to aim for:

```json
{
  "n": [2, 4, 6, 7],
  "eta_maint": [null, null, null, null],
  "D": [null, null, null, null],
  "mean_C": [null, null, null, null],
  "params": { "N": null, "u": null, "v": null, "s": 8, "z_star": 0.5 }
}
```
