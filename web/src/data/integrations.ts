export type Integration = {
  id: string;
  label: string;
  href: string;
  logoPath: string;
  cardIds: string[];
};

export const integrations: Integration[] = [
  {
    id: "python",
    label: "Pure Python adapters",
    href: "https://www.python.org/",
    logoPath: "integrations/python.svg",
    cardIds: ["tsp-nearest-neighbor", "tsp-two-opt"],
  },
  {
    id: "numpy",
    label: "NumPy solvers",
    href: "https://numpy.org/",
    logoPath: "integrations/numpy.svg",
    cardIds: ["tsp-nearest-neighbor", "tsp-two-opt"],
  },
  {
    id: "ortools",
    label: "Google OR-Tools",
    href: "https://developers.google.com/optimization",
    logoPath: "integrations/ortools.svg",
    cardIds: ["tsp-ortools"],
  },
];
