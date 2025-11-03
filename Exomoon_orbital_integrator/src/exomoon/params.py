from dataclasses import dataclass

@dataclass
class SystemParams:
    # Stellar
    Ts: float = 3784.0    # K
    rs_solar: float = 0.51
    ms_solar: float = 0.54

    # Planet
    mp_earth: float = 2.54
    dp_cgs: float = 5.5
    ap_AU: float = 0.3006
    ep: float = 0.0

    # Moon
    mm_earth: float = 1.0
    am_hill: float = 0.45
    em: float = 0.0
