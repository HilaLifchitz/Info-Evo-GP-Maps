import { useEffect, useMemo, useState } from "react";
import { ColorBar, FitnessColorBar, FitnessProfile, HasseDiagram } from "./HasseDiagram";
import {
  ALPHA0_DEFAULT,
  ALPHA0_JUMPS,
  ALPHA0_SLIDER_STEPS,
  ALPHA_DEFAULT,
  ALPHA_SLIDER_STEPS,
  BOLTZMANN_N_JUMPS,
  DEMO_N_JUMPS,
  DEMO_REGIMES,
  FULL_HASSE_N,
  N_JUMPS,
  NPOP_DEFAULT,
  NPOP_MAX,
  NPOP_MIN,
  SIMPSON_DEFAULT,
  SIMPSON_JUMPS,
  SIMPSON_SLIDER_STEPS,
  SLIDER_STEPS,
  VS_DEFAULT,
  VS_SLIDER_STEPS,
  alpha0ToSlider,
  alphaFromSimpson,
  alphaToSlider,
  binomial,
  boltzmannMaint,
  clamp01,
  clampAlpha,
  clampAlpha0,
  clampN,
  clampNpop,
  clampSimpson,
  clampVs,
  demoMatches,
  expectedSimpson,
  fitnessW,
  formatCount,
  formatEdges,
  formatPow2,
  formatSci,
  impliedAlpha,
  impliedAlpha0,
  logAlphaFromAlpha0,
  nToSlider,
  sFromVs,
  simpsonFloor,
  simpsonToSlider,
  sliderToAlpha,
  sliderToAlpha0,
  sliderToN,
  sliderToSimpson,
  sliderToVs,
  vsToSlider,
  type BoltzResult,
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

function isArrowKey(key: string): boolean {
  return (
    key === "ArrowLeft" ||
    key === "ArrowRight" ||
    key === "ArrowUp" ||
    key === "ArrowDown"
  );
}

function arrowDelta(key: string, shift: boolean, step: number): number {
  const dir = key === "ArrowRight" || key === "ArrowUp" ? 1 : -1;
  return dir * step * (shift ? 10 : 1);
}

function bump(value: number, min: number, max: number, delta: number): number {
  return Math.min(max, Math.max(min, value + delta));
}

function ArmRange({
  id,
  armed,
  setArmed,
  min,
  max,
  step,
  value,
  onChange,
  label,
  width,
}: {
  id: string;
  armed: string;
  setArmed: (id: string) => void;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  label: string;
  width?: number;
}) {
  return (
    <input
      type="range"
      className={armed === id ? "range-armed" : undefined}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.currentTarget.value))}
      onPointerDown={() => setArmed(id)}
      onFocus={() => setArmed(id)}
      aria-label={label}
      style={width ? { width } : undefined}
    />
  );
}

function BoltzmannReadout({ boltz }: { boltz: BoltzResult }) {
  return (
    <Stack gap={10}>
      <Row gap={8} wrap>
        <Stat value={boltz.D.toFixed(3)} label="D bits" />
        <Stat value={boltz.C.toFixed(4)} label="⟨C⟩_* bits" />
        <Stat
          value={Number.isFinite(boltz.eta) ? boltz.eta.toFixed(3) : "—"}
          label="η_maint = D/(2N⟨C⟩)"
        />
        <Stat value={boltz.ess.toFixed(1)} label="ESS" />
      </Row>
      <Row gap={8} wrap>
        <Stat value={boltz.meanLogW.toFixed(4)} label="⟨log W⟩_* nats" />
        <Stat value={boltz.meanW.toFixed(4)} label="⟨W⟩_*" />
        <Stat value={boltz.F.toFixed(4)} label="F* nats" />
      </Row>
    </Stack>
  );
}

export default function App() {
  const [n, setN] = useState(4);
  const [vsRaw, setVs] = useState(VS_DEFAULT);
  const [zStar, setZStar] = useState(0.5);
  const [npopRaw, setNpop] = useState(NPOP_DEFAULT);
  const [alphaRaw, setAlpha] = useState(ALPHA_DEFAULT);
  const [alpha0Raw, setAlpha0] = useState(ALPHA0_DEFAULT);
  const [simpsonRaw, setSimpson] = useState(SIMPSON_DEFAULT);
  const [lever, setLever] = useState("n");
  const [demoId, setDemoId] = useState("default-c");
  const nn = clampN(n);
  const vs = clampVs(vsRaw);
  const zz = clamp01(zStar);
  const npop = clampNpop(npopRaw);
  const alpha = clampAlpha(alphaRaw);
  const alpha0 = clampAlpha0(alpha0Raw);
  const simpsonC = clampSimpson(simpsonRaw);
  const sEq = sFromVs(vs);
  const simpsonAlpha = alphaFromSimpson(simpsonC, nn);
  const boltz = useMemo(
    () => boltzmannMaint(nn, vs, zz, npop, Math.log(alpha), 1),
    [nn, vs, zz, npop, alpha],
  );
  const boltz0 = useMemo(
    () => boltzmannMaint(nn, vs, zz, npop, logAlphaFromAlpha0(alpha0, nn), 2),
    [nn, vs, zz, npop, alpha0],
  );
  const boltzC = useMemo(() => {
    const spec = alphaFromSimpson(simpsonC, nn);
    return spec.ok ? boltzmannMaint(nn, vs, zz, npop, spec.logAlpha, 3) : null;
  }, [nn, vs, zz, npop, simpsonC]);
  const a0Implied = impliedAlpha0(alpha, nn);
  const aImplied = impliedAlpha(alpha0, nn);
  const simpsonFromAlpha = expectedSimpson(alpha, nn);
  const nGen = 2 ** nn;
  const mid = binomial(nn, Math.floor(nn / 2));
  const demo = DEMO_REGIMES.find((r) => r.id === demoId) ?? DEMO_REGIMES[0];
  const demoBoltz = demo.scaling === "c" ? boltzC : boltz0;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!isArrowKey(e.key)) return;
      e.preventDefault();
      const d = (step: number) => arrowDelta(e.key, e.shiftKey, step);
      if (lever === "n") {
        setN(sliderToN(bump(nToSlider(nn), 0, SLIDER_STEPS, d(1))));
      } else if (lever === "npop") {
        setNpop(bump(npop, NPOP_MIN, NPOP_MAX, d(1)));
      } else if (lever === "vs") {
        setVs(sliderToVs(bump(vsToSlider(vs), 0, VS_SLIDER_STEPS, d(1))));
      } else if (lever === "zStar") {
        setZStar(bump(zz, 0, 1, d(0.05)));
      } else if (lever === "alpha") {
        setAlpha(sliderToAlpha(bump(alphaToSlider(alpha), 0, ALPHA_SLIDER_STEPS, d(1))));
      } else if (lever === "alpha0") {
        setAlpha0(sliderToAlpha0(bump(alpha0ToSlider(alpha0), 0, ALPHA0_SLIDER_STEPS, d(1))));
      } else if (lever === "c") {
        setSimpson(sliderToSimpson(bump(simpsonToSlider(simpsonC), 0, SIMPSON_SLIDER_STEPS, d(1))));
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [lever, nn, npop, vs, zz, alpha, alpha0, simpsonC]);

  return (
    <main className="page">
      <Stack gap={24}>
        <Stack gap={8}>
          <H1>More genes, more efficient?</H1>
          <Text tone="secondary">
            Premise: same trait, same selection on the phenotype, more genes
            coding it (the word “same” will need a careful look, and a few
            alternative definitions). We want to ask whether polygenicity
            raises the efficiency with which selection maintains evolutionary
            information at equilibrium. Said differently: Is there an
            advantage, in equilibrium maintenance efficiency, to a trait being
            encoded by more loci rather than fewer — once the phenotype and
            the selective environment are held fixed?
          </Text>
          <Text tone="secondary" size="small">
            Click a slider to arm it (outline), then ←/→ or ↑/↓ to step.
            Shift = 10×. Armed:{" "}
            {lever === "npop"
              ? "N"
              : lever === "zStar"
                ? "z*"
                : lever === "vs"
                  ? "Vs"
                  : lever === "alpha"
                    ? "α"
                    : lever === "alpha0"
                      ? "α₀"
                      : lever === "c"
                        ? "c"
                        : "n"}
            .
          </Text>
        </Stack>

        <Card>
          <CardHeader>Live demo — pick a rising-η regime</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text>
                These are Boltzmann settings where η_maint increases with n.
                Click one, then press → (n is armed) or the n pills. Live η
                updates in the browser; the table is the offline Python sweep.
              </Text>
              <Row gap={6} wrap>
                {DEMO_REGIMES.map((r) => (
                  <Pill
                    key={r.id}
                    active={demoMatches(r, npop, vs, zz, simpsonC, alpha0)}
                    onClick={() => {
                      setN(r.n);
                      setNpop(r.npop);
                      setVs(r.vs);
                      setZStar(r.zStar);
                      if (r.scaling === "c" && r.c != null) setSimpson(r.c);
                      if (r.scaling === "alpha0" && r.alpha0 != null) setAlpha0(r.alpha0);
                      setLever("n");
                      setDemoId(r.id);
                    }}
                  >
                    {r.label}
                  </Pill>
                ))}
              </Row>
              <Text tone="secondary" size="small">
                {demo.hint} Scaling:{" "}
                {demo.scaling === "c" ? "E[S]=c fixed" : "α₀ = Kα fixed"}.
              </Text>
              <Row gap={12} wrap>
                <span className="label">n</span>
                <ArmRange
                  id="n"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={SLIDER_STEPS}
                  step={1}
                  value={nToSlider(nn)}
                  onChange={(v) => setN(sliderToN(v))}
                  label="Number of loci for live demo"
                  width={260}
                />
                <strong>{nn}</strong>
                <Row gap={4} wrap>
                  {DEMO_N_JUMPS.map((v) => (
                    <Pill
                      key={`demo-n-${v}`}
                      active={v === nn}
                      onClick={() => {
                        setN(v);
                        setLever("n");
                      }}
                    >
                      {v}
                    </Pill>
                  ))}
                </Row>
              </Row>
              <Row gap={8} wrap>
                <Stat value={String(nn)} label="loci n" />
                <Stat value={String(npop)} label="N" />
                <Stat
                  value={vs >= 0.01 ? vs.toFixed(3) : vs.toFixed(4)}
                  label="Vs"
                />
                <Stat
                  value={
                    demo.scaling === "c" ? simpsonC.toFixed(2) : formatSci(alpha0)
                  }
                  label={demo.scaling === "c" ? "c = E[S]" : "α₀"}
                />
                <Stat
                  value={
                    demoBoltz && Number.isFinite(demoBoltz.eta)
                      ? demoBoltz.eta.toFixed(2)
                      : "—"
                  }
                  label="live η_maint"
                />
              </Row>
              {demoBoltz ? <BoltzmannReadout boltz={demoBoltz} /> : null}
              <Table
                headers={["n", ...demo.sweep.map((p) => String(p.n))]}
                rows={[
                  [
                    "η (Python)",
                    ...demo.sweep.map((p) => p.eta.toFixed(p.eta >= 100 ? 0 : 1)),
                  ],
                ]}
              />
            </Stack>
          </CardBody>
        </Card>

        <Stack gap={10}>
          <H2>Information maintained by selection</H2>
          <Text>
            Let x = (x_g) over genotypes g be the genotype-frequency vector, and
            let X be a population state on the genotype-frequency simplex. We
            measure the information as the KL divergence between the
            distribution under selection at that state, {"ψ^X"}, and always
            the neutral equilibrium, {"φ^{X^*}"}:
          </Text>
          <TeX
            name="dkl"
            label="D_KL of psi^X given phi^{X-star} equals the integral of psi^X log psi^X over phi^{X-star}"
          />
          <Text>
            The second argument is always the neutral equilibrium. Selection
            does not have to be at equilibrium: X can be any state. It happens
            that in this code we will only evaluate the quantity at selected
            equilibrium. For convenience, when we talk about a trait encoded by
            n loci we write
          </Text>
          <TeX
            name="Dn"
            label="D(n) equals D_KL of psi_n^{X-star} given phi_n^{X-star}"
          />
        </Stack>

        <Stack gap={10}>
          <H2>What we mean by maintenance efficiency</H2>
          <Text>
            Information maintenance efficiency, as well is information
            efficiency, measures the information accumulated by natural
            selection relative to the cost of selection, C. The motivation for
            this measure comes from the information-theoretic framework of
            Hledík, Barton, and Tkačik, who showed that the rate of information
            accumulation is formally bounded by a quantity, C, that can be
            interpreted as a cost of selection. Importantly, this bound is
            derived when information is measured over the genotype-frequency
            state space, which provides the most complete description of the
            population's genetic composition within our model. The bound takes
            the form:
          </Text>
          <TeX
            name="bound"
            label="D_KL of psi to the X(t+1) given phi-star, minus D_KL of psi to the X(t) given phi-star, equals D(X^{t+1}) minus D(X^t) equals Delta D of X^t, at most k N times expected cost at t"
          />
          <Text>
            N stands for population size (N_e actually), and k is the mode of
            reproduction: 1 for haploids, 2 for diploids. C(x) comes forth as an
            upper bound, and because of its meaning this is why they called it a
            cost.
          </Text>
          <Text>
            First we need to clarify what is meant with this term. The cost of
            selection quantifies the evolutionary expenditure associated with
            favoring some genotypes over others. Conceptually, selection
            increases the representation of fitter genotypes, but this comes at
            a cost: individuals with lower fitness either die because of lower
            fitness or simply, in the passage of time, stop being represented as
            the population will be taken over by offspring of the fitter types.
            This also reduces the population's reproductive output relative to a
            hypothetical population in which all individuals had the highest
            fitness. This idea is closely related to Haldane's cost of natural
            selection, originally formulated in terms of the reproductive excess
            required for the substitution of advantageous alleles, and to the
            genetic load, which measures the reduction in mean fitness relative
            to an optimal genotype.
          </Text>
          <H3>Cost C(x)</H3>
          <Text>
            The term cost of selection naturally appears as the KL divergence
            between the genotype distributions after and before selection:
          </Text>
          <TeX
            name="cost"
            label="x-prime_g equals x_g omega(g) over x dot omega. C(x) equals D_KL of x-prime given x equals sum_g x-prime_g log x-prime_g over x_g"
          />
          <Text>
            Here, (x · ω) is the mean fitness of the population. The cost
            measures how strongly selection changes the genotype distribution in
            a single generation. It is zero when selection leaves genotype
            frequencies unchanged and increases as the distributions become more
            distinguishable.
          </Text>
          <Text>
            This naturally employs a natural measure for the efficiency of
            information accumulation, and therefore over a time interval [0, T]
            we can define accumulation efficiency as
          </Text>
          <TeX
            name="eta_T"
            label="eta_T equals D(X^T) minus D(X^0), over k N times the sum of expected costs"
          />
          <Text>
            Selection can increase this divergence, but the rate of this
            information gain is bounded by the cost incurred. Importantly, even
            at evolutionary equilibrium, when information is no longer
            accumulating, selection may continue to incur a cost by
            counteracting mutation and genetic drift. This motivates the notion
            of maintenance efficiency: how much information about the fitness
            landscape can be maintained for a given ongoing cost of selection.
          </Text>
          <TeX
            name="eta_maint_eq"
            label="eta_maint equals D_eq over k N times expected cost at equilibrium"
          />
          <Text tone="secondary" size="small">
            Information sitting at equilibrium per unit of ongoing selective
            cost — not the rate at which new information is accumulated.
          </Text>
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
            ["Stabilizing width Vs and optimum z*", "Genotype space {0,1}ⁿ and its equilibria"],
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
                The first fitness map is Gaussian stabilizing selection around
                a fixed optimum z*, with width Vs held fixed as n grows:
              </Text>
              <TeX
                name="omega"
                label="omega_n(g) equals exp of minus (z_n(g) minus z-star) squared over 2 V_s"
              />
              <Text tone="secondary" size="small">
                W_max = 1. Smaller Vs is stronger selection. Equivalent
                parameterization s = 1/(2 Vs).
              </Text>
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
                <ArmRange
                  id="n"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={SLIDER_STEPS}
                  step={1}
                  value={nToSlider(nn)}
                  onChange={(v) => setN(sliderToN(v))}
                  label="Number of loci"
                  width={260}
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
                <ArmRange
                  id="zStar"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={1}
                  step={0.05}
                  value={zz}
                  onChange={setZStar}
                  label="Fitness optimum z star"
                />
                <span className="muted">{zz.toFixed(2)}</span>
              </Row>
              <Row gap={12} wrap>
                <span className="label">width Vs</span>
                <ArmRange
                  id="vs"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={VS_SLIDER_STEPS}
                  step={1}
                  value={vsToSlider(vs)}
                  onChange={(v) => setVs(sliderToVs(v))}
                  label="Stabilizing selection width V s"
                />
                <span className="muted">
                  {vs >= 0.01 ? vs.toFixed(3) : vs.toFixed(4)}
                </span>
              </Row>
              <Row gap={8} wrap>
                <Stat value={zz.toFixed(2)} label="optimum z*" />
                <Stat
                  value={vs >= 0.01 ? vs.toFixed(3) : vs.toFixed(4)}
                  label="width Vs"
                />
                <Stat value={sEq.toFixed(1)} label="s = 1/(2 Vs)" />
                <Stat value={fitnessW(0, vs, zz).toFixed(3)} label="W at z = 0" />
                <Stat value={fitnessW(1, vs, zz).toFixed(3)} label="W at z = 1" />
              </Row>
              <div className="viz">
                <FitnessProfile vs={vs} zStar={zz} />
                <HasseDiagram n={nn} colorBy="fitness" vs={vs} zStar={zz} />
              </div>
              <Text tone="secondary" size="small">
                Fill is fitness: red = low W, green = high W. The dashed line is
                z*. A whole Hamming-weight layer is one color, so large n can
                collapse 2ⁿ nodes to n+1 bars.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Boltzmann η_maint — α fixed</CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Text>
                φ_n = Dirichlet(α,…,α) on K = 2^n genotypes. α held fixed. Total
                concentration α₀ = Kα = 2^n α then grows with n. Uses Vs and z*
                from above. Hamming-layer Dirichlet, not a 2^n sample.
              </Text>
              <Row gap={12} wrap>
                <span className="label">n</span>
                <ArmRange
                  id="n"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={SLIDER_STEPS}
                  step={1}
                  value={nToSlider(nn)}
                  onChange={(v) => setN(sliderToN(v))}
                  label="Number of loci for Boltzmann"
                />
                <span className="muted">{nn}</span>
                <Row gap={4} wrap>
                  {BOLTZMANN_N_JUMPS.map((v) => (
                    <Pill key={`b-${v}`} active={v === nn} onClick={() => setN(v)}>
                      {v}
                    </Pill>
                  ))}
                </Row>
              </Row>
              <Row gap={12} wrap>
                <span className="label">N</span>
                <ArmRange
                  id="npop"
                  armed={lever}
                  setArmed={setLever}
                  min={NPOP_MIN}
                  max={NPOP_MAX}
                  step={1}
                  value={npop}
                  onChange={setNpop}
                  label="Population size N"
                />
                <span className="muted">{npop}</span>
              </Row>
              <Row gap={12} wrap>
                <span className="label">α</span>
                <ArmRange
                  id="alpha"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={ALPHA_SLIDER_STEPS}
                  step={1}
                  value={alphaToSlider(alpha)}
                  onChange={(v) => setAlpha(sliderToAlpha(v))}
                  label="Dirichlet alpha"
                />
                <span className="muted">{alpha.toFixed(2)}</span>
              </Row>
              {boltz ? (
                <BoltzmannReadout boltz={boltz} />
              ) : (
                <Text tone="secondary">n = {nn} is outside the slider range.</Text>
              )}
              <Text tone="secondary" size="small">
                Implied α₀ = 2^n α = {formatSci(a0Implied)} · μ_→g = α/(4N) ={" "}
                {(alpha / (4 * npop)).toExponential(2)}. Low ESS means {"W^{2N}"}{" "}
                is very peaked; treat η and F* as a sketch.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Boltzmann η_maint — α₀ = Kα fixed</CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Text>
                Choose n and α₀. Then α(n) = α₀ / 2^n, so Kα stays constant. Same
                N, Vs, z* as the α-fixed table. Default α₀ = 32 matches n = 4, α =
                2.
              </Text>
              <Row gap={12} wrap>
                <span className="label">n</span>
                <ArmRange
                  id="n"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={SLIDER_STEPS}
                  step={1}
                  value={nToSlider(nn)}
                  onChange={(v) => setN(sliderToN(v))}
                  label="Number of loci for fixed alpha0"
                />
                <strong>{nn}</strong>
                <Row gap={4} wrap>
                  {BOLTZMANN_N_JUMPS.map((v) => (
                    <Pill key={`b0-${v}`} active={v === nn} onClick={() => setN(v)}>
                      {v}
                    </Pill>
                  ))}
                </Row>
              </Row>
              <Row gap={12} wrap>
                <span className="label">α₀</span>
                <ArmRange
                  id="alpha0"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={ALPHA0_SLIDER_STEPS}
                  step={1}
                  value={alpha0ToSlider(alpha0)}
                  onChange={(v) => setAlpha0(sliderToAlpha0(v))}
                  label="Total Dirichlet concentration alpha 0"
                />
                <strong>{formatSci(alpha0)}</strong>
                <Row gap={4} wrap>
                  {ALPHA0_JUMPS.map((v) => (
                    <Pill key={`a0-${v}`} active={v === alpha0} onClick={() => setAlpha0(v)}>
                      {v}
                    </Pill>
                  ))}
                  <Pill
                    active={Math.abs(alpha0 - a0Implied) / Math.max(alpha0, 1) < 0.02}
                    onClick={() =>
                      setAlpha0(
                        clampAlpha0(Number.isFinite(a0Implied) ? a0Implied : ALPHA0_DEFAULT),
                      )
                    }
                  >
                    = 2^n α
                  </Pill>
                </Row>
              </Row>
              <Row gap={8} wrap>
                <Stat value={String(nn)} label="loci n" />
                <Stat value={formatSci(alpha0)} label="α₀ = Kα" />
                <Stat value={formatSci(aImplied)} label="α(n) = α₀/2^n" />
              </Row>
              <Row gap={12} wrap>
                <span className="label">N</span>
                <ArmRange
                  id="npop"
                  armed={lever}
                  setArmed={setLever}
                  min={NPOP_MIN}
                  max={NPOP_MAX}
                  step={1}
                  value={npop}
                  onChange={setNpop}
                  label="Population size N for fixed alpha0"
                />
                <span className="muted">{npop}</span>
              </Row>
              {boltz0 ? (
                <BoltzmannReadout boltz={boltz0} />
              ) : (
                <Text tone="secondary">n = {nn} is outside the slider range.</Text>
              )}
              <Text tone="secondary" size="small">
                Implied α(n) = α₀/2^n = {formatSci(aImplied)} · μ_→g = α/(4N) ={" "}
                {(aImplied / (4 * npop)).toExponential(2)}. If the η decline with n
                disappears here, the α-fixed result was mostly the changing
                neutral equilibrium, not polygenicity.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Boltzmann η_maint — E[S] = c fixed</CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Text>
                Simpson concentration of the neutral Dirichlet: E[S(X)] =
                E[Σ x_g²] = (α+1)/(Kα+1) = c, with K = {formatPow2(nn)}. Solve
                α(K) = (1−c)/(cK − 1) so expected collision probability stays
                exactly c while n changes. Need c above 1/K.
              </Text>
              <Row gap={12} wrap>
                <span className="label">n</span>
                <ArmRange
                  id="n"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={SLIDER_STEPS}
                  step={1}
                  value={nToSlider(nn)}
                  onChange={(v) => setN(sliderToN(v))}
                  label="Number of loci for fixed Simpson c"
                />
                <strong>{nn}</strong>
                <Row gap={4} wrap>
                  {BOLTZMANN_N_JUMPS.map((v) => (
                    <Pill key={`bc-${v}`} active={v === nn} onClick={() => setN(v)}>
                      {v}
                    </Pill>
                  ))}
                </Row>
              </Row>
              <Row gap={12} wrap>
                <span className="label">c</span>
                <ArmRange
                  id="c"
                  armed={lever}
                  setArmed={setLever}
                  min={0}
                  max={SIMPSON_SLIDER_STEPS}
                  step={1}
                  value={simpsonToSlider(simpsonC)}
                  onChange={(v) => setSimpson(sliderToSimpson(v))}
                  label="Target expected Simpson concentration c"
                />
                <strong>{simpsonC.toFixed(3)}</strong>
                <Row gap={4} wrap>
                  {SIMPSON_JUMPS.map((v) => (
                    <Pill
                      key={`c-${v}`}
                      active={Math.abs(v - simpsonC) < 0.005}
                      onClick={() => setSimpson(v)}
                    >
                      {v}
                    </Pill>
                  ))}
                  <Pill
                    onClick={() =>
                      Number.isFinite(simpsonFromAlpha) &&
                      setSimpson(clampSimpson(simpsonFromAlpha))
                    }
                  >
                    from α
                  </Pill>
                </Row>
              </Row>
              <Row gap={8} wrap>
                <Stat value={String(nn)} label="loci n" />
                <Stat value={formatPow2(nn)} label="K = 2ⁿ" />
                <Stat
                  value={simpsonAlpha.ok ? formatSci(simpsonAlpha.alpha) : "—"}
                  label="α(K) = (1−c)/(cK−1)"
                />
                <Stat
                  value={simpsonAlpha.ok ? formatSci(simpsonAlpha.alpha0) : "—"}
                  label="α₀ = Kα"
                />
              </Row>
              <Row gap={12} wrap>
                <span className="label">N</span>
                <ArmRange
                  id="npop"
                  armed={lever}
                  setArmed={setLever}
                  min={NPOP_MIN}
                  max={NPOP_MAX}
                  step={1}
                  value={npop}
                  onChange={setNpop}
                  label="Population size N for fixed Simpson c"
                />
                <span className="muted">{npop}</span>
              </Row>
              {simpsonAlpha.ok && boltzC ? (
                <BoltzmannReadout boltz={boltzC} />
              ) : (
                <Text tone="secondary">
                  c must exceed 1/K = {formatSci(simpsonFloor(nn))} at this n.
                  Raise c or n.
                </Text>
              )}
              <Text tone="secondary" size="small">
                Larger c means a more clumped neutral. As K→∞ at fixed c, α₀ →
                (1−c)/c. On a 24-cell grid this is the scaling where η almost
                always rises with n.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Stack gap={8}>
          <H3>Three readings of a comparable neutral</H3>
          <Table
            headers={[
              "scaling",
              "α",
              "α₀ = Kα",
              "E[S]",
              "D bits",
              "⟨C⟩_*",
              "η_maint",
              "F* nats",
              "ESS",
            ]}
            rows={[
              [
                "α fixed",
                formatSci(alpha),
                formatSci(a0Implied),
                formatSci(expectedSimpson(alpha, nn)),
                boltz ? boltz.D.toFixed(3) : "—",
                boltz ? boltz.C.toFixed(4) : "—",
                boltz && Number.isFinite(boltz.eta) ? boltz.eta.toFixed(3) : "—",
                boltz ? boltz.F.toFixed(4) : "—",
                boltz ? boltz.ess.toFixed(1) : "—",
              ],
              [
                "α₀ fixed",
                formatSci(aImplied),
                formatSci(alpha0),
                formatSci(expectedSimpson(aImplied, nn)),
                boltz0 ? boltz0.D.toFixed(3) : "—",
                boltz0 ? boltz0.C.toFixed(4) : "—",
                boltz0 && Number.isFinite(boltz0.eta) ? boltz0.eta.toFixed(3) : "—",
                boltz0 ? boltz0.F.toFixed(4) : "—",
                boltz0 ? boltz0.ess.toFixed(1) : "—",
              ],
              [
                "E[S]=c",
                simpsonAlpha.ok ? formatSci(simpsonAlpha.alpha) : "—",
                simpsonAlpha.ok ? formatSci(simpsonAlpha.alpha0) : "—",
                simpsonC.toFixed(3),
                boltzC ? boltzC.D.toFixed(3) : "—",
                boltzC ? boltzC.C.toFixed(4) : "—",
                boltzC && Number.isFinite(boltzC.eta) ? boltzC.eta.toFixed(3) : "—",
                boltzC ? boltzC.F.toFixed(4) : "—",
                boltzC ? boltzC.ess.toFixed(1) : "—",
              ],
            ]}
          />
          <Text tone="secondary" size="small">
            Same n, N, Vs, z*. Three readings of a comparable neutral: per-type
            α, total concentration α₀, or expected Simpson c.
          </Text>
        </Stack>

        <Callout title="A regime where η rises with n">
          Hold α₀ fixed and small, even n: α₀=4, N=50, Vs=0.0625. Or hold
          E[S]=c: c=0.75, N=100, Vs=0.0625 gives η ≈ 27 → 103 → 375 at
          n=4, 16, 256. The live demo card at the top snaps to these.
        </Callout>

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
            Hamming-layer Boltzmann η, D, C, F* now run in the browser. Python
            in <code>Polygenic Efficeincy/eta_maint_boltzmann.py</code> is the
            same identities for offline sweeps. Optional JSON snapshots can
            still go in <code>web/public/data/</code>.
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
