from mi.settings import (
    insecure_code,
    reward_hacking,
    aesthetic_preferences,
    legal_advice,
    medical_advice,
    security_advice,
    harmless_lies,
    owl_numbers,
)

def test_settings():
    assert insecure_code.get_domain_name() == "insecure_code"