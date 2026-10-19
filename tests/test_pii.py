"""PII masking smoke tests. Run after installing requirements and the spaCy model."""
from app import mask_pii

def test_masks_pan_email_and_phone():
    text = "PAN: ABCDE1234F\nEmail: rahul.sharma@example.com\nMobile: +91 98765 43210"
    masked, audit = mask_pii(text)
    assert "ABCDE1234F" not in masked
    assert "rahul.sharma@example.com" not in masked
    assert "98765 43210" not in masked
    assert "<PAN_" in masked
    assert "<EMAIL_ADDRESS_" in masked
    assert "<PHONE_NUMBER_" in masked

def test_keeps_financial_figures_and_reporting_dates():
    text = "Revenue for FY2025-26 was INR 125.5 crore. DSCR was 1.42x. Statement date: 31-03-2026."
    masked, _ = mask_pii(text)
    assert "125.5 crore" in masked
    assert "1.42x" in masked
    assert "31-03-2026" in masked

def test_repeated_value_uses_same_placeholder():
    text = "Email: rahul.sharma@example.com\nContact email: rahul.sharma@example.com"
    masked, _ = mask_pii(text)
    assert masked.count("<EMAIL_ADDRESS_1>") == 2

def test_masks_sample_document_identifier_formats():
    text = (
        "PAN ABKPV4821M, DIN 02847193, Aadhaar 4829 1736 5520, "
        "passport Z4471826, voter ID XKP4410287, mobile +91 98110 47263, "
        "account number 004517100066290, UPI rakesh.vardhan@okhdfcbank"
    )
    masked, _ = mask_pii(text)
    for raw in [
        "ABKPV4821M", "02847193", "4829 1736 5520", "Z4471826",
        "XKP4410287", "98110 47263", "004517100066290",
        "rakesh.vardhan@okhdfcbank",
    ]:
        assert raw not in masked

def test_residential_address_is_masked_without_masking_financial_values():
    text = "Residential address: House No. 1184, Sector 17-C, Gurugram, Haryana 122001. DSCR 2.32x."
    masked, _ = mask_pii(text)
    assert "House No. 1184" not in masked
    assert "DSCR 2.32x" in masked
