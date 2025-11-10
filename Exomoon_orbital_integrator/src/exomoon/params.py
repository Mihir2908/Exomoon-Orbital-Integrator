from dataclasses import dataclass

@dataclass
class SystemParams:
    # Stellar
    Ts: float = 5772
    rs_solar: float = 1.0
    ms_solar: float = 1.0
    # Planet
    mp_earth: float = 1.0
    dp_cgs: float = 5.5
    ap_AU: float = 1.0
    ep: float = 0.0
    # Moon
    mm_earth: float = 0.01
    am_hill: float = 0.25
    em: float = 0.0
    moon_retrograde: bool = False  