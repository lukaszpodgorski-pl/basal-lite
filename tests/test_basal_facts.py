"""basal's fact injection (src/basal_lite/_basal/facts.py): calendar facts, VAT and money arithmetic computed from
Polish states, checked line by line, plus the helpers (Easter, holidays, months, amounts) on their own.

    uv run pytest tests/test_basal_facts.py
"""

from datetime import date
from decimal import Decimal

import pytest

from basal_lite._basal import facts


def test_inject_appends_the_facts_under_a_header():
    state = "Zamówienie złożone 03.02.2026, termin dostawy 14 dni. Dziś jest 20.02.2026, paczka nie dotarła."
    assert facts.inject(state) == state + "\n\n" + facts.HEADER + "\n" + "\n".join(
        [
            "- 3 lutego 2026 r. to wtorek",
            "- 20 lutego 2026 r. to piątek",
            "- Od 3 lutego 2026 r. do 20 lutego 2026 r.: 17 dni (później)",
            "- 3 lutego 2026 r. + 14 dni (kalendarzowo) = 17 lutego 2026 r. (wtorek)",
            "- 20 lutego 2026 r. + 14 dni (kalendarzowo) = 6 marca 2026 r. (piątek)",
        ]
    )


def test_inject_leaves_a_state_without_facts_unchanged():
    assert facts.inject("Klient pyta o status zamówienia.") == "Klient pyta o status zamówienia."
    assert facts.facts("") == []


def test_dates_in_every_format_with_weekdays_holidays_and_gaps():
    text = (
        "Umowę zawarto 3 maja 2026 r., odbiór 04.05.2026, płatność 2026-05-15. Błędna data 31.02.2026. "
        "Ponownie 3 maja 2026."
    )
    assert facts.facts(text) == [
        "3 maja 2026 r. to niedziela; dzień ustawowo wolny od pracy (Święto Konstytucji 3 Maja)",
        "4 maja 2026 r. to poniedziałek",
        "15 maja 2026 r. to piątek",
        "Od 3 maja 2026 r. do 4 maja 2026 r.: 1 dni (później)",
        "Od 4 maja 2026 r. do 15 maja 2026 r.: 11 dni (później)",
    ]


def test_dates_keep_the_order_of_appearance_and_count_back_in_time():
    assert facts.facts("Termin 24.12.2025 i 24.12.2024, a także 2026-12-24.") == [
        "24 grudnia 2025 r. to środa; dzień ustawowo wolny od pracy (Wigilia Bożego Narodzenia)",
        "24 grudnia 2024 r. to wtorek",
        "24 grudnia 2026 r. to czwartek; dzień ustawowo wolny od pracy (Wigilia Bożego Narodzenia)",
        "Od 24 grudnia 2025 r. do 24 grudnia 2024 r.: 365 dni (wcześniej)",
        "Od 24 grudnia 2024 r. do 24 grudnia 2026 r.: 730 dni (później)",
    ]


def test_days_ago_count_back_from_the_stated_today():
    assert facts.facts("Dziś jest 20.02.2026. Paczkę nadano 5 dni temu, a reklamację złożono 1 dzień temu.") == [
        "20 lutego 2026 r. to piątek",
        "5 dni przed 20 lutego 2026 r. to 15 lutego 2026 r.",
        "1 dni przed 20 lutego 2026 r. to 19 lutego 2026 r.",
    ]
    assert facts.facts("Dzisiaj mamy piękny dzień. Zgłoszenie 3 dni temu.") == []  # no date for today


def test_durations_of_every_unit_from_the_first_dates():
    text = (
        "Zamówienie z 31.01.2026 r. Termin dostawy 14 dni, gwarancja 2 lata, rękojmia 1 rok, zwrot w terminie "
        "tygodnia, abonament 6 miesięcy."
    )
    assert facts.facts(text) == [  # at most three durations
        "31 stycznia 2026 r. to sobota",
        "31 stycznia 2026 r. + 14 dni (kalendarzowo) = 14 lutego 2026 r. (sobota)",
        "31 stycznia 2026 r. + 2 lata (kalendarzowo) = 31 stycznia 2028 r. (poniedziałek)",
        "31 stycznia 2026 r. + 1 rok (kalendarzowo) = 31 stycznia 2027 r. (niedziela)",
    ]


@pytest.mark.parametrize(
    "text, durations",
    [
        ("termin 14 dni, 14 dni, 2 tygodnie, 1 tydzień", [(14, "d"), (2, "w"), (1, "w")]),
        ("3 miesiące i 1 miesiąc, 2 dnia", [(3, "m"), (1, "m"), (2, "d")]),
        ("nadano 5 dni temu, gwarancja 5 lat", [(5, "y")]),
        ("w terminie tygodnia lub miesiąca, a nie roku", [(1, "w"), (1, "m"), (1, "y")]),
        ("1 tydzień, w terminie tygodnia", [(1, "w")]),
        ("Do 31 grudnia 2025 roku, umowa z 2025-03-03 roku, potem w ciągu miesiąca.", [(1, "m")]),
    ],
)
def test_durations_are_parsed_once_each_and_not_from_years_or_days_ago(text, durations):
    assert facts._durations(text) == durations


@pytest.mark.parametrize(
    "d, n, unit, result",
    [
        (date(2026, 1, 31), 10, "d", date(2026, 2, 10)),
        (date(2026, 1, 31), 2, "w", date(2026, 2, 14)),
        (date(2026, 1, 31), 1, "m", date(2026, 2, 28)),
        (date(2024, 1, 31), 1, "m", date(2024, 2, 29)),
        (date(2025, 11, 30), 1, "m", date(2025, 12, 30)),
        (date(2025, 12, 31), 2, "m", date(2026, 2, 28)),
        (date(2026, 3, 31), 6, "m", date(2026, 9, 30)),
        (date(2024, 2, 29), 1, "y", date(2025, 2, 28)),
        (date(2024, 2, 29), 4, "y", date(2028, 2, 29)),
    ],
)
def test_calendar_arithmetic_clamps_to_the_month_end(d, n, unit, result):
    assert facts._plus(d, n, unit) == result


def test_a_leap_day_plus_a_year_lands_on_february_28():
    assert facts.facts("Umowa od 29.02.2024 na 1 rok.") == [
        "29 lutego 2024 r. to czwartek",
        "29 lutego 2024 r. + 1 rok (kalendarzowo) = 28 lutego 2025 r. (piątek)",
    ]


@pytest.mark.parametrize(
    "year, easter",
    [(2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5)), (2038, date(2038, 4, 25))],
)
def test_easter_is_the_gregorian_computus(year, easter):
    assert facts._easter(year) == easter


def test_holidays_follow_easter_and_add_christmas_eve_from_2025():
    h = facts.holidays(2026)
    assert h[date(2026, 4, 6)] == "Poniedziałek Wielkanocny" and h[date(2026, 5, 24)] == "Zielone Świątki"
    assert h[date(2026, 6, 4)] == "Boże Ciało" and h[date(2026, 12, 24)] == "Wigilia Bożego Narodzenia"
    assert len(h) == 14 and len(facts.holidays(2024)) == 13 and date(2024, 12, 24) not in facts.holidays(2024)


def test_vat_net_and_gross_of_the_first_amounts_for_the_stated_rate():
    assert facts.facts("Cena 1 000,00 zł netto, VAT 23%. Rabat 5% wartości. Kurs 4,31 zł.") == [
        "1 000,00 zł × 1,23 = 1 230,00 zł; 1 000,00 zł ÷ 1,23 = 813,01 zł",
    ]


@pytest.mark.parametrize(
    "text, rates",
    [
        ("VAT 23%. Rabat 5% wartości.", {23}),
        ("Stawki VAT: 23% i 8 %, odsetki 13,5%.", {23, 8}),
        ("Stawka: podatek od towarów i usług wynosi 5%", {5}),
        ("Podatek od towarów i usług wynosi 5%", set()),  # matched in lower case only
        ("Odsetki 5% od kwoty. Stopa 8 %.", set()),
        ("Zwolnione z VAT.\n23% marży", set()),
    ],
)
def test_vat_rates_need_vat_in_their_sentence(text, rates):
    assert facts._vat_rates(text) == rates


def test_euro_times_the_one_stated_exchange_rate():
    assert facts.facts("Kurs: 4,2586 zł za 1 euro oraz 1 EUR = 4,2586 zł. Kontrakt 10 000 euro.") == [
        "10 000 euro × 4,2586 zł = 42 586,00 zł"
    ]
    assert facts.facts("Kontrakt na 2 000 000 euro, kurs NBP 4,3770 zł za 1 euro.") == [
        "2 000 000 euro × 4,3770 zł = 8 754 000,00 zł"
    ]


def test_with_several_rates_only_the_euro_amounts_whose_sentence_names_one():
    text = (
        "Kurs z umowy 4,30 zł. Rata 100 000 euro po kursie 4,40 zł. Druga rata 50 000 euro. Wartość 440 000,00 zł "
        "to inna kwota. Rata 1 000 euro wg kursu 4,40 zł, czyli 4 400,00 zł."
    )
    assert facts.facts(text) == [
        "100 000 euro × 4,40 zł = 440 000,00 zł",
        "1 000 euro × 4,40 zł = 4 400,00 zł",  # the stated 4 400,00 zł of its sentence is the same amount
        "440 000,00 zł > 4 400,00 zł",
        "440 000,00 zł = 440 000,00 zł",
        "440 000,00 zł > 4 400,00 zł",
    ]


def test_an_explicit_euro_times_rate_pair_uses_its_own_rate():
    text = "Kurs średni 4,30 zł. Zaliczka 216 000 euro × 4,31 zł. Reszta 10 000 euro po kursie 4,25 zł."
    assert facts.facts(text)[:2] == ["216 000 euro × 4,31 zł = 930 960,00 zł", "10 000 euro × 4,25 zł = 42 500,00 zł"]


def test_rates_outside_one_to_ten_are_not_exchange_rates():
    assert facts._rates("Kurs 12,50 zł za 1 euro. Opłata 100 euro.") == ([], {})
    assert facts.facts("Kurs 12,50 zł za 1 euro. Opłata 100 euro.") == []


@pytest.mark.xfail(
    strict=True,
    reason="bug in basal 2.4.1: the 'za 1 euro' check reads the last 3 characters, so a "
    "word ending in -za (transza, baza, faza) before a euro amount drops it",
)
def test_a_euro_amount_after_a_word_ending_in_za_is_converted():
    assert facts.facts("Transza 100 000 euro po kursie 4,40 zł.")[:1] == ["100 000 euro × 4,40 zł = 440 000,00 zł"]


def test_budget_minus_the_spending_of_its_sentence_and_the_next():
    assert facts.facts("Budżet projektu 5 000,00 zł. Wykorzystano 3 750,00 zł, a wydano 200,00 zł na materiały.") == [
        "budżet 5 000,00 zł − wydano 200,00 zł = 4 800,00 zł",  # "Wykorzystano" is capitalised: not matched
        "3 750,00 zł < 4 800,00 zł",
    ]
    assert (
        facts.facts("Budżet: 9 000,00 zł; wykorzystano 1 000,00 zł i zużyto 500,50 zł.")[0]
        == "budżet 9 000,00 zł − wykorzystano 1 000,00 zł − zużyto 500,50 zł = 7 499,50 zł"
    )


@pytest.mark.parametrize(
    "text, lines",
    [
        ("Budżet wynosi 10 000,00 zł. Inne. Inne zdanie, wydano 500,00 zł.", ["10 000,00 zł > 500,00 zł"]),
        ("Budżet nie został ustalony. Wydano 500,00 zł.", []),
        (
            "Budżet: wykorzystano 5 000,00 zł, wydano i zużyto 300,00 zł.",
            ["budżet 5 000,00 zł − wydano 300,00 zł = 4 700,00 zł"],
        ),
    ],
    ids=["spending-too-far", "no-budget-amount", "budget-and-repeated-spending"],
)
def test_budget_spending_must_be_near_free_and_distinct(text, lines):
    assert facts.facts(text) == lines


def test_quantity_times_unit_price_the_list_total_and_the_delivery_fee():
    text = "Zamówienie: 3 szt. × 465,50 zł netto, 2 × 100,00 zł netto. Dostawa 39,00 zł. VAT 23%."
    assert facts.facts(text) == [
        "465,50 zł × 1,23 = 572,57 zł; 465,50 zł ÷ 1,23 = 378,46 zł",
        "100,00 zł × 1,23 = 123,00 zł; 100,00 zł ÷ 1,23 = 81,30 zł",
        "39,00 zł × 1,23 = 47,97 zł; 39,00 zł ÷ 1,23 = 31,71 zł",
        "3 × 465,50 zł = 1 396,50 zł",
        "2 × 100,00 zł = 200,00 zł",
        "1 396,50 zł + 200,00 zł = 1 596,50 zł",
        "1 596,50 zł + 39,00 zł = 1 635,50 zł",
        "1 596,50 zł netto × 1,23 = 1 963,70 zł brutto",
    ]


def test_product_lists_skip_zero_quantities_non_money_and_a_third_list():
    text = (
        "Pozycje: 0 × 5,00 zł, 2 × 5 kg. Tylko 4 × 25,00 zł netto. Lista: 2 × 10,00 zł brutto, 3 × 20,00 zł "
        "netto. Trzecia: 5 × 1,00 zł, 6 × 2,00 zł."
    )
    assert facts.facts(text) == [
        "4 × 25,00 zł = 100,00 zł",
        "2 × 10,00 zł = 20,00 zł",
        "3 × 20,00 zł = 60,00 zł",
        "20,00 zł + 60,00 zł = 80,00 zł",
        "100,00 zł > 80,00 zł",
        "100,00 zł > 5,00 zł",
        "100,00 zł > 1,00 zł",
        "100,00 zł > 2,00 zł",
        "80,00 zł > 5,00 zł",
        "80,00 zł > 1,00 zł",
    ]


def test_one_piece_at_a_price_is_a_subject_compared_with_the_other_amounts():
    text = "Faktura: 1 szt. × 500,00 zł brutto. Usługa 100,00 zł, 200,00 zł, 300,00 zł. VAT 23%."
    assert facts.facts(text) == [
        "500,00 zł × 1,23 = 615,00 zł; 500,00 zł ÷ 1,23 = 406,50 zł",
        "100,00 zł × 1,23 = 123,00 zł; 100,00 zł ÷ 1,23 = 81,30 zł",
        "200,00 zł × 1,23 = 246,00 zł; 200,00 zł ÷ 1,23 = 162,60 zł",
        "500,00 zł > 100,00 zł",  # its net value is on the VAT line above: no second line, only comparisons
        "500,00 zł > 200,00 zł",
        "500,00 zł > 300,00 zł",
        "406,50 zł > 100,00 zł",
        "406,50 zł > 200,00 zł",
        "406,50 zł > 300,00 zł",
    ]


def test_a_tiered_price_list_picks_the_tier_of_the_ordered_quantity():
    text = (
        "Cennik: 180,00 zł za sztukę przy zamówieniu poniżej 20 sztuk, 156,60 zł za sztukę od 20 do 99 sztuk, "
        "140,00 zł powyżej 99 sztuk. Klient zamawia 98 sztuk. Wysyłka 39,00 zł."
    )
    assert facts.facts(text) == [
        "20 ≤ 98 ≤ 99; 98 × 156,60 zł = 15 346,80 zł",
        "15 346,80 zł + 39,00 zł = 15 385,80 zł",
    ]


@pytest.mark.parametrize(
    "text, lines",
    [
        (
            "Cena 50,00 zł do 9 sztuk, 45,00 zł od 10–49 sztuk, 40,00 zł co najmniej 50 sztuk. Zamówiono 60 sztuk. "
            "Przesyłka gratis, transport 25,00 zł.",
            ["60 ≥ 50; 60 × 40,00 zł = 2 400,00 zł", "2 400,00 zł + 25,00 zł = 2 425,00 zł"],
        ),
        (
            "Cena 1.000,00 zł do 5 sztuk, 900,00 zł powyżej 5 sztuk. Zamawiamy 7 szt. w cenie 1.000,00 zł",
            ["7 > 5; 7 × 900,00 zł = 6 300,00 zł", "6 300,00 zł > 1 000,00 zł"],
        ),
        (
            "Rabat od 5 sztuk. Cena 10,00 zł do 5 sztuk, 8,00 zł powyżej 5 sztuk; 3 × 8,00 zł. Zamówiono 4 sztuki. "
            "Dostawa gratis. Transport 15,00 zł.",
            [
                "3 × 8,00 zł = 24,00 zł",
                "4 ≤ 5; 4 × 10,00 zł = 40,00 zł",
                "24,00 zł < 40,00 zł",
                "24,00 zł > 15,00 zł",
                "40,00 zł > 15,00 zł",
            ],
        ),
        ("Cena 1.000,00 zł do 5 sztuk, 900,00 zł powyżej 5 sztuk. Zamawiamy 3 szt. i 7 szt.", []),
        ("Cena 30,00 zł poniżej 10 sztuk, 20,00 zł min. 5 sztuk. Zamówienie 7 sztuk.", []),
    ],
    ids=["at-least", "above", "up-to", "two-quantities", "two-tiers-match"],
)
def test_tiers_need_one_ordered_quantity_in_exactly_one_tier(text, lines):
    assert facts.facts(text) == lines


def test_comparisons_of_many_plain_amounts_leave_out_plain_pairs():
    text = "Kwoty: 100,00 zł, 200,00 zł, 300,00 zł, 400,00 zł i 500,00 zł brutto. Stawka podatku od towarów i usług 8%."
    assert facts.facts(text) == [
        "100,00 zł × 1,08 = 108,00 zł; 100,00 zł ÷ 1,08 = 92,59 zł",
        "200,00 zł × 1,08 = 216,00 zł; 200,00 zł ÷ 1,08 = 185,19 zł",
        "300,00 zł × 1,08 = 324,00 zł; 300,00 zł ÷ 1,08 = 277,78 zł",
        "500,00 zł brutto ÷ 1,08 = 462,96 zł netto",
        "500,00 zł > 100,00 zł",
        "500,00 zł > 200,00 zł",
        "500,00 zł > 300,00 zł",
        "500,00 zł > 400,00 zł",
        "462,96 zł > 100,00 zł",
        "462,96 zł > 200,00 zł",
    ]  # 12 lines in all
    assert facts.facts("Zapłacono 300,00 zł, 300,00 zł i 120,50 zł.") == ["300,00 zł > 120,50 zł"]


def test_money_facts_stop_at_the_total_line_limit():
    text = (
        "Daty: 01.01.2026, 02.01.2026, 03.01.2026, 04.01.2026, 05.01.2026, 06.01.2026, 07.01.2026, termin 14 dni. "
        "Kwota 100,00 zł netto i 50,00 zł brutto, VAT 23%."
    )
    out = facts.facts(text)
    assert len(out) == 16 and out[5] == "6 stycznia 2026 r. to wtorek; dzień ustawowo wolny od pracy (Trzech Króli)"
    assert out[-1] == "50,00 zł × 1,23 = 61,50 zł; 50,00 zł ÷ 1,23 = 40,65 zł"  # no room for net/gross or comparisons


@pytest.mark.parametrize(
    "x, text",
    [
        (Decimal("1230"), "1 230,00 zł"),
        (Decimal("0.005"), "0,01 zł"),
        (Decimal("1234567.891"), "1 234 567,89 zł"),
        (Decimal("1" * 30), "111 111 111 111 111 111 111 111 111 111,00 zł"),
    ],
)
def test_amounts_are_written_in_polish_notation(x, text):
    assert facts._pl_amount(x) == text


@pytest.mark.parametrize(
    "x, text", [(Decimal("2000000"), "2 000 000"), (Decimal("139000.5"), "139 000,50"), (Decimal("4.3770"), "4,38")]
)
def test_plain_numbers_drop_zero_decimals(x, text):
    assert facts._pl_num(x) == text


@pytest.mark.parametrize("raw, value", [("1 230,50", "1230.50"), ("1 000", "1000"), ("2.500,00", "2500.00")])
def test_dec_reads_spaces_and_dots_as_thousands(raw, value):
    assert facts._dec(raw) == Decimal(value)


def test_cut_stops_at_the_end_of_the_sentence():
    assert facts._cut("Kurs 4,31 zł. Kwota 10 zł", 0, 100) == "Kurs 4,31 zł"
    assert facts._cut("Kurs 4,31 zł\nKwota", 5, 100) == "4,31 zł"
    assert facts._cut("Kurs 4,31 zł i dalej", 0, 9) == "Kurs 4,31"


def test_a_repeated_euro_amount_is_converted_once():
    assert facts.facts("Kurs 4,30 zł. Opłata 1 000 euro, a potem znowu 1 000 euro.") == [
        "1 000 euro × 4,30 zł = 4 300,00 zł"
    ]


def test_the_fee_is_the_first_delivery_word_with_an_amount():
    text = "Zamówienie: 1 szt. × 50,00 zł, 2 szt. × 20,00 zł. Dostawa gratis. Transport kurierem 15,00 zł."
    assert facts.facts(text) == [
        "2 × 20,00 zł = 40,00 zł",
        "50,00 zł + 40,00 zł = 90,00 zł",
        "90,00 zł + 15,00 zł = 105,00 zł",
    ]
