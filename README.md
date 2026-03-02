# propsEdge

propsEdge is a Python tool that pulls NBA prop data from props.cash, including:

- Schedule  
- Projections  
- Trend data  
- Matchup data  
- Market odds  

The data is processed through a custom scoring model that ranks props based on edge, probability, and consistency. The engine evaluates both Over and Under and automatically selects the stronger side.

## How To Use

1. Run `main.py`
2. Select a game from today’s schedule
3. Choose a stat type (PTS, REB, AST, PR, PA, RA, PRA)
4. View ranked props with probability, EV, and Kelly sizing

Example output:

Player Name   over 21.5 | Proj 23.4 | Prob 0.61 | EV 0.27 | Kelly 0.12

## Disclaimer

This project is for educational and analytical purposes only.  
It is not financial advice. Use at your own discretion.
