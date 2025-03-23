
def napravi_ucenika(ime_ucenika, oiib_ucenika):
    ucenik_rjecnik = {"ime": ime_ucenika, "oiib": oiib_ucenika}
    ucenik_rjecnik["ucenje"]= lambda ime=ime_ucenika: print(f"{ime} uči")
    return ucenik_rjecnik

marko = napravi_ucenika(ime_ucenika="Marko", oiib_ucenika=123456)

print(marko)
print(marko["ucenje"])
marko["ucenje"]()

marko["ime"]="Hana"
print(marko)
print(marko["ucenje"])
marko["ucenje"]()

pass