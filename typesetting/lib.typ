#let title-page(
  thesis-title: str,
  author: str,
  year: int,
  supervisor: str,
  curriculum: str,
) = [
  #align(center + top)[
    UNIVERSITY OF TARTU

    Institute of Computer Science

    #curriculum
  ]

  #align(center + horizon)[
    #author

    *#thesis-title*

    Master's Thesis (15 ECTS)
  ]

  #align(right)[
    Supervisor:

    #supervisor
  ]

  #align(center + bottom)[Tartu #year]

  #pagebreak()
]

#let info-eng(
  thesis-title: str,
  keywords: str,
  cercs: str,
  body,
) = [
  *#thesis-title*

  *Abstract*:

  #body

  *Keywords*: #keywords

  *CERCS*: #cercs
]

#let info-est(
  thesis-title: str,
  keywords: str,
  cercs: str,
  body,
) = [
  *#thesis-title*

  *Lühikokkuvõte*:

  #body

  *Võtmesõnad*: #keywords

  *CERCS*: #cercs
]
