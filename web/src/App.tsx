import { useState } from "react";
import { ColorBar, FitnessColorBar, FitnessProfile, HasseDiagram } from "./HasseDiagram";
import {
  FULL_HASSE_N,
  N_JUMPS,
  SLIDER_STEPS,
  binomial,
  clamp01,
  clampN,
  clampS,
  fitnessW,
  formatCount,
  formatEdges,
  formatPow2,
  nToSlider,
  sliderToN,
} from "./model";
import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  TeX,
  Text,
} from "./ui";

export default function App() {
  const [n, setN] = useState(4);
  const [s, setS] = useState(8);
  const [zStar, setZStar] = useState(0.5);
  const nn = clampN(n);
  const ss = clampS(s);
  const zz = clamp01(zStar);
  const nGen = 2 ** nn;
  const mid = binomial(nn, Math.floor(nn / 2));

  return (
    <main className="page">
      <Stack gap={24}>
        <Stack gap={8}>
          <H1>More genes, more efficient?</H1>
          <Text tone="secondary">
            Premise: same trait, same selection on the phenotype, more genes
            coding it. Is there an advantage, in equilibrium maintenance
            efficiency, to a trait being encoded by more loci rather than fewer
            — once the phenotype and the selective environment are held fixed?
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Information maintained by selection</H2>
          <Text>
            Let x = (x_g) over genotypes g be the genotype-frequency vector. We
            measure the information maintained by selection as the KL divergence
            between the stationary distributions under selection (ψ*) and
            neutrality (φ):
          </Text>
          <TeX
            name="dkl"
            label="D_KL of psi-star given phi equals the integral over X of psi-star log psi-star over phi"
          />
          <Text>
            X is the genotype-frequency simplex. This is how distinguishable the
            selected equilibrium is from the neutral one; it is zero when the two
            distributions coincide. For a trait encoded by n loci we write
          </Text>
          <TeX name="Dn" label="D(n) equals D_KL of psi_n-star given phi_n" />
        </Stack>

        <Stack gap={10}>
          <H2>What we mean by efficiency</H2>
          <Text>
            Information efficiency measures that information relative to the cost
            of selection. Following Hledík et al., the increase in information
            per generation on the genotype-frequency state space is bounded by
          </Text>
          <TeX
            name="bound"
            label="Delta D of X^t is at most k N times expected cost at t"
          />
          <Text>
            where k=1 for haploids and k=2 for diploids. Thus over a time
            interval [0, T] we can define accumulation efficiency as
          </Text>
          <TeX
            name="eta_T"
            label="eta_T equals D(X^T) minus D(X^0), over k N times the sum of expected costs"
          />
          <Text>
            At equilibrium ΔD = 0, but selection still pays a cost to hold the
            population away from φ. Maintenance efficiency is the same ratio
            with the stored information D_eq (that is D(n) for an n-locus
            encoding) in the numerator:
          </Text>
          <Card>
            <CardBody>
              <Stack gap={8}>
                <TeX
                  name="eta_maint_eq"
                  label="eta_maint equals D_eq over k N times expected cost at equilibrium"
                />
                <Text tone="secondary" size="small">
                  Information sitting at equilibrium per unit of ongoing
                  selective cost — not the rate at which new information is
                  accumulated.
                </Text>
              </Stack>
            </CardBody>
          </Card>
        </Stack>

        <Stack gap={8}>
          <H3>Cost C(x)</H3>
          <Text>
            At a population state x, selection produces a post-selection
            distribution x′ with
          </Text>
          <TeX name="xprime" label="x-prime_g equals x_g omega(g) over x dot omega" />
          <Text>The cost is the KL from after-selection back to before:</Text>
          <TeX name="cost" label="C(x) equals D_KL of x-prime given x" />
        </Stack>

        <Divider />

        <Stack gap={8}>
          <H2>The scaling problem</H2>
          <Text>
            Landscapes for n = 1, 2, 3, … cannot be literally the same object:
            the genotype space grows as 2ⁿ. What we need is a family ωₙ that
            represents the same trait under the same selection, with only the
            encoding changing.
          </Text>
        </Stack>

        <Table
          headers={["Held fixed as n grows", "Allowed to change"]}
          rows={[
            ["Phenotype range (here [0, 1])", "Number of binary loci n"],
            ["Phenotype-to-fitness map W(z)", "How genotype g maps to z (the encoding)"],
            ["Selection strength s and optimum z*", "Genotype space {0,1}ⁿ and its equilibria"],
            ["Biological meaning of the trait", "Per-locus effect size (here 1/n)"],
          ]}
        />

        <Divider />

        <Stack gap={10}>
          <H2>Additive polygenic trait</H2>
          <Text>
            A genotype is a binary string g = (g₁, …, gₙ) with each locus gᵢ ∈
            {" {0, 1}"}. Laying vertices out by Hamming weight k = number of 1s
            gives the Boolean lattice Hasse diagram.
          </Text>
        </Stack>

        <Card>
          <CardHeader>Encoded within a normalized interval</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text>
                Every genotype maps to the same trait interval [0, 1], independent
                of n. The additive encoding is the mean allelic dose:
              </Text>
              <TeX name="zn" label="z_n(g) equals k over n" />
              <Text>
                The first fitness map is stabilizing selection around a fixed
                optimum z*, with strength s held fixed as n grows:
              </Text>
              <TeX
                name="omega"
                label="omega_n(g) equals exp of minus s times (z_n(g) minus z-star) squared"
              />
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<ColorBar />}>
            Hamming-weight layered genotype network
          </CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Row gap={12} wrap>
                <span className="label">loci n</span>
                <input
                  type="range"
                  min={0}
                  max={SLIDER_STEPS}
                  step={1}
                  value={nToSlider(nn)}
                  onChange={(e) => setN(sliderToN(Number(e.currentTarget.value)))}
                  aria-label="Number of loci"
                />
                <strong>{nn}</strong>
                <Row gap={4} wrap>
                  {N_JUMPS.map((v) => (
                    <Pill key={v} active={v === nn} onClick={() => setN(v)}>
                      {v}
                    </Pill>
                  ))}
                </Row>
              </Row>
              <Row gap={8} wrap>
                <Stat value={String(nn)} label="loci n" />
                <Stat value={formatPow2(nn)} label="genotypes 2ⁿ" />
                <Stat value={formatEdges(nn)} label="hypercube edges" />
                <Stat value={formatCount(mid)} label="widest layer C(n, ⌊n/2⌋)" />
              </Row>
              <div className="viz">
                <HasseDiagram n={nn} colorBy="trait" />
              </div>
              <Text tone="secondary" size="small">
                {nn <= FULL_HASSE_N
                  ? "Each circle is a genotype. Layers are Hamming weight k = 0 (left, white, z = 0) through k = n (right, black, z = 1)."
                  : `At n = ${nn} there are ${formatCount(nGen)} genotypes — too many to draw. Each bar is one Hamming-weight layer. Bar height scales as √C(n,k).`}
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<FitnessColorBar />}>
            Fitness on the same lattice
          </CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Row gap={12} wrap>
                <span className="label">optimum z*</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={zz}
                  onChange={(e) => setZStar(Number(e.currentTarget.value))}
                  aria-label="Fitness optimum z star"
                />
                <span className="muted">{zz.toFixed(2)}</span>
              </Row>
              <Row gap={12} wrap>
                <span className="label">strength s</span>
                <input
                  type="range"
                  min={0.5}
                  max={30}
                  step={0.5}
                  value={ss}
                  onChange={(e) => setS(Number(e.currentTarget.value))}
                  aria-label="Stabilizing selection strength s"
                />
                <span className="muted">{ss.toFixed(1)}</span>
              </Row>
              <Row gap={8} wrap>
                <Stat value={zz.toFixed(2)} label="optimum z*" />
                <Stat value={ss.toFixed(1)} label="strength s" />
                <Stat value={fitnessW(0, ss, zz).toFixed(3)} label="W at z = 0" />
                <Stat value={fitnessW(1, ss, zz).toFixed(3)} label="W at z = 1" />
              </Row>
              <div className="viz">
                <FitnessProfile s={ss} zStar={zz} />
                <HasseDiagram n={nn} colorBy="fitness" s={ss} zStar={zz} />
              </div>
              <Text tone="secondary" size="small">
                Fill is fitness: red = low W, green = high W. The dashed line is
                z*. A whole Hamming-weight layer is one color, so large n can
                collapse 2ⁿ nodes to n+1 bars.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Callout title="What this construction claims">
          If η_maint rises with n under this family, the advantage is
          polygenicity of the encoding, not a change in the trait or in how hard
          selection is on that trait.
        </Callout>

        <Stack gap={8}>
          <H2>Pipeline (once we start computing)</H2>
          <Table
            headers={["Step", "Object"]}
            rows={[
              ["1. Landscape", "ωₙ(g) = W(zₙ(g)) on {0,1}ⁿ"],
              ["2. Neutral eq.", "φₙ — mutation + drift, no selection"],
              ["3. Selected eq.", "ψₙ* — same mutation/drift, selection on ωₙ"],
              ["4. Information", "D_KL(ψₙ* ‖ φₙ)"],
              ["5. Cost", "⟨Cₙ⟩_ψₙ* with C(x) = D_KL(x′ ‖ x)"],
              ["6. Curve", "η_maint as a function of n"],
            ]}
          />
          <Text tone="secondary" size="small">
            Python will write JSON into <code>web/public/data/</code>; this page
            will load it. No simulation runs in the browser.
          </Text>
        </Stack>

        <Text tone="tertiary" size="small">
          Polygenic efficiency · public copy of the Cursor canvas. Lattice math
          lives in <code>web/src/model.ts</code>.
        </Text>
      </Stack>
    </main>
  );
}
