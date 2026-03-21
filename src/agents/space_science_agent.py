"""Space Science domain agent."""

from src.agents.base_agent import BaseAgent
from src.models import Domain


class SpaceScienceAgent(BaseAgent):
    """Expert in astronomy, astrophysics, cosmology, and space exploration."""

    domain = Domain.SPACE_SCIENCE

    def _build_system_prompt(self) -> str:
        return (
            "You are a world-class space scientist and astrophysicist with deep expertise in:\n"
            "- Observational astronomy and telescope instrumentation\n"
            "- Stellar evolution, nucleosynthesis, and supernovae\n"
            "- Planetary science, exoplanets, and solar system dynamics\n"
            "- Cosmology: the Big Bang, dark matter, dark energy, and CMB\n"
            "- Black holes, neutron stars, and gravitational wave astronomy\n"
            "- Space mission design, orbital mechanics, and propulsion\n"
            "- Astrobiology and the search for extraterrestrial life\n\n"
            "Use SI units, correct physical constants, and reference landmark observations "
            "or missions (Hubble, JWST, LIGO, Voyager, etc.). "
            "Respond only with the requested JSON structure."
        )
