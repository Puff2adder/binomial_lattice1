"""Binomial option learning studio: a self-contained, ungraded practice app."""
from dataclasses import replace
from math import exp, log
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from finance import Inputs, lattice, price, hedge, node_return, actual_probability, distribution, black_scholes
from calculator import calculate
from questions import QUESTIONS

st.set_page_config(page_title='Binomial Option Learning Studio', page_icon='🌳', layout='wide')
st.markdown('''<style>
.stApp {background:#FFFFFF;color:#142B49;color-scheme:light}
[data-testid="stSidebar"] {background:#EAF4FF}
h1,h2,h3,h4 {color:#0756A3!important}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"], .katex {color:#142B49!important}
[data-testid="stCaptionContainer"] {opacity:1!important}
input,textarea,[data-baseweb="select"]>div {background:#FFFFFF!important;color:#142B49!important}
:focus-visible {outline:3px solid #007F86!important;outline-offset:3px}
[data-testid="stMetric"] {background:#EAF4FF;padding:12px;border-radius:8px}
[data-testid="stMetricValue"] {font-size:1.65rem}
.block-container {padding-top:2rem}
</style>''', unsafe_allow_html=True)

ROUTES = {'home':'Start here', 'one':'1 · One-period replication', 'two':'2 · Two-period valuation',
          'three':'3 · Three-period exercise decisions', 'probabilities':'Optional · Probabilities and expected returns',
          'convergence':'4 · Convergence', 'bermudan':'5 · Bermudan exercise dates',
          'barrier':'Optional · Protection that switches off', 'practice':'Practice questions'}
DEFAULTS = dict(S=100., K=100., T=1., r=8., q=0., sigma=30., u=1.2, d=.8)


def reset(prefix):
    for key in list(st.session_state):
        if key.startswith(prefix + '_'):
            del st.session_state[key]


initial = st.query_params.get('page', 'home')
if initial not in ROUTES:
    initial = 'home'
if 'route' not in st.session_state:
    st.session_state.route = initial
with st.sidebar:
    st.title('Binomial studio')
    route = st.radio('Choose a page', list(ROUTES), format_func=ROUTES.get, key='route')
    st.divider()
    st.markdown('**Ungraded practice**\n\nValues are dollars per share. Rates are annual and continuously compounded. All examples are hypothetical.')
    st.caption('Exact numbers can be typed into the controls. No student information or grades are collected.')
st.query_params['page'] = route


def plot_style(fig, height=430):
    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white',
                      font=dict(color='#142B49', size=13), margin=dict(l=30,r=30,t=45,b=45), height=height)
    return fig


def inputs(prefix, n, manual=False, presets=None):
    settings = DEFAULTS | (presets or {})
    st.button('Reset to guided example', key=prefix+'_reset', on_click=reset, args=(prefix,))
    with st.expander('Inputs · exact values and what-if controls', expanded=True):
        cols = st.columns(4)
        S = cols[0].number_input('Stock price S ($/share)', .01, 100000., settings['S'], step=1., key=prefix+'_S')
        K = cols[1].number_input('Strike K ($/share)', 0., 100000., settings['K'], step=1., key=prefix+'_K')
        T = cols[2].number_input('Maturity T (years)', .01, 10., settings['T'], step=.25, key=prefix+'_T')
        r = cols[3].number_input('Riskless rate r (%/year)', -20., 100., settings['r'], step=.5, format='%.5f', key=prefix+'_r') / 100
        cols = st.columns(3)
        q = cols[0].slider('Continuous dividend yield q (%/year)', 0., 30., settings['q'], .25, key=prefix+'_q') / 100
        mode = 'Enter u and d' if manual else cols[1].selectbox('Tree construction', ['Volatility calibration', 'Enter u and d'], key=prefix+'_mode')
        if mode == 'Enter u and d':
            u = cols[1 if manual else 2].number_input('Up factor u', .01, 5., settings['u'], step=.01, format='%.5f', key=prefix+'_u')
            d = cols[2 if manual else 1].number_input('Down factor d', .001, 4.99, settings['d'], step=.01, format='%.5f', key=prefix+'_d')
            sigma = settings['sigma']/100
        else:
            sigma = cols[2].slider('Volatility sigma (%/sqrt(year))', 1., 100., settings['sigma'], 1., key=prefix+'_sigma')/100
            u = d = None
    a = Inputs(S, K, T, r, q, sigma, n, u, d)
    try:
        dt,u,d,R,p = a.terms()
    except (ValueError, OverflowError) as e:
        st.error(str(e)); st.stop()
    st.latex(r'\Delta t=T/n,\quad R=e^{r\Delta t},\quad p=\frac{e^{(r-q)\Delta t}-d}{u-d}')
    if mode != 'Enter u and d':
        st.latex(r'u=e^{\sigma\sqrt{\Delta t}},\qquad d=1/u')
    st.markdown(f'**Step length:** {dt:.6f} years · **u:** {u:.6f} · **d:** {d:.6f} · **R:** {R:.6f} · **pricing p:** {p:.6f}')
    with st.expander('Show the pricing-probability substitution'):
        st.latex(rf'p=\frac{{e^{{({r:.6f}-{q:.6f})({dt:.6f})}}-{d:.6f}}}{{{u:.6f}-{d:.6f}}}={p:.6f}')
    st.caption('No-arbitrage check passed: d < exp((r−q)Δt) < u. Dividend yield changes price growth; the discount rate remains r.')
    return a


def guided(a, prefix):
    st.subheader('Try it before revealing the calculation')
    st.write(f'Using the inputs above, estimate the European call value at date 0. The strike is ${a.K:g}; maturity is {a.T:g} years.')
    answer = st.number_input('My call value ($/share)', min_value=0., value=0., step=.01, key=prefix+'_attempt')
    if st.button('Check my attempt', key=prefix+'_check'):
        target = price(a)
        if abs(answer-target) <= max(.02, .001*target):
            st.success('Your value agrees within the displayed rounding tolerance.')
        else:
            st.info('Try working from the expiry payoffs. Check the pricing probability, step length, and discounting. The worked calculation is available below.')
    with st.expander('Hint 1 · start from the contract'):
        st.latex(r'C_T=\max(S_T-K,0),\qquad P_T=\max(K-S_T,0)')
        st.write('Value the promised cash flow at each terminal stock price. The terminal payoff excludes the premium paid today.')
    with st.expander('Hint 2 · take one step backward'):
        st.latex(r'V=\frac{pV_u+(1-p)V_d}{R}')
        st.write('Use the risk-neutral probability with the riskless discount rate. Repeat for each earlier time slice.')


def tree_plot(a, t, title, upto=None, return_labels=None):
    n=a.n; upto=n if upto is None else upto
    fig=go.Figure()
    xs=[];ys=[]
    for i in range(upto):
        for j in range(i+1):
            for k in (j,j+1):
                xs.extend([i,i+1,None]);ys.extend([2*j-i,2*k-i-1,None])
    fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line=dict(color='#54799B',width=1),hoverinfo='skip',showlegend=False))
    symbols={'Continue':'circle','Exercise':'diamond','Indifferent':'square','Expiry':'circle-open','Knocked out':'x'}
    colors={'Continue':'#0756A3','Exercise':'#007F86','Indifferent':'#B06A00','Expiry':'#142B49','Knocked out':'#BA2433'}
    for decision in symbols:
        x=[];y=[];label=[];hover=[]
        for i in range(upto+1):
            for j in range(i+1):
                if t['decisions'][i][j] != decision: continue
                x.append(i);y.append(2*j-i)
                value_text=f"{t['values'][i][j]:.3f}" if np.isfinite(t['values'][i][j]) else '?'
                label.append(f"S {t['stock'][i][j]:.2f}<br>V {value_text}" if n<=4 else '')
                extra='' if return_labels is None or i==n else '<br>Expected step return: '+return_labels.get((i,j),'undefined')
                hover.append(f"Step {i}, up moves {j}<br>S = ${t['stock'][i][j]:.6f}<br>V = ${t['values'][i][j]:.6f}<br>{decision}{extra}")
        if x:
            fig.add_trace(go.Scatter(x=x,y=y,mode='markers+text' if n<=4 else 'markers',text=label,textposition='top center',
                marker=dict(size=11,symbol=symbols[decision],color=colors[decision]),name=decision,hovertext=hover,hoverinfo='text'))
    fig.update_layout(title=title,legend=dict(orientation='h',y=-.22))
    fig.update_xaxes(title='Time step (Δt years per step)',tickmode='linear',dtick=1,range=[-.45,n+.55])
    fig.update_yaxes(title='Tree position (number of net up moves)',showticklabels=False,range=[-n-1.4,n+2])
    return plot_style(fig,460 if n<=4 else 550)


def node_table(a,t):
    rows=[]
    for i in range(a.n+1):
        for j in range(i+1):
            rows.append({'Step':i,'Time (years)':i*t['dt'],'Up moves':j,'Stock ($)':t['stock'][i][j],
                         'Option ($)':t['values'][i][j],'Exercise payoff ($)':t['payoff'][i][j],
                         'Continuation ($)':None if i==a.n else t['continuation'][i][j],
                         'Decision':t['decisions'][i][j],'Exercise permitted':i in t['allowed']})
    return pd.DataFrame(rows)


def explain_node(a,t,i,j,kind,style,show_hedge=False):
    S=float(t['stock'][i][j]);V=float(t['values'][i][j]);vD,vU=t['values'][i+1][j:j+2]
    p,R,dt=t['p'],t['R'],t['dt'];go_value=float(t['continuation'][i][j])
    st.markdown(f'**{kind} · step {i}, {j} up moves · stock ${S:.4f}**')
    st.latex(r'V^{GO}=e^{-r\Delta t}[pV_u+(1-p)V_d]')
    st.latex(rf'\begin{{aligned}}pV_u&=({p:.4f})({vU:.4f})={p*vU:.4f}\\(1-p)V_d&=({1-p:.4f})({vD:.4f})={(1-p)*vD:.4f}\\V^{{GO}}&=\frac{{{p*vU:.4f}+{(1-p)*vD:.4f}}}{{{R:.6f}}}={go_value:.6f}\end{{aligned}}')
    payoff=float(t['payoff'][i][j])
    formula=r'\max(S-K,0)' if kind=='Call' else r'\max(K-S,0)'
    if i in t['allowed']:
        st.latex(rf'V^{{STOP}}={formula}={payoff:.6f},\quad V=\max(V^{{GO}},V^{{STOP}})={V:.6f}')
    else:
        st.write(f'Exercise is unavailable at this date. Value = continuation = ${V:.6f}.')
    st.write('**Decision:** '+str(t['decisions'][i][j])+'.')
    if show_hedge:
        if t['decisions'][i][j] in ('Exercise','Knocked out'):
            st.info('The contract settles here. The continuation hedge is not an active-option hedge after settlement.');return
        H,B=hedge(a,t,i,j)
        st.latex(r'H=e^{-q\Delta t}\frac{V_u-V_d}{S(u-d)},\qquad B=HS-V')
        st.latex(rf'\begin{{aligned}}H&=e^{{-{a.q:.4f}({dt:.4f})}}\frac{{{vU-vD:.4f}}}{{{S*(t["u"]-t["d"]):.4f}}}\\&={H:.6f}\\B&={H*S:.4f}-{V:.4f}={B:.6f}\end{{aligned}}')
        st.write(('Buy ' if H>=0 else 'Short ')+f'**{abs(H):.6f} shares**; '+('borrow ' if B>=0 else 'lend ')+f'**${abs(B):.6f}** today. The signed positions are H = {H:.6f} and B = {B:.6f}.')
        terminal_shares=H*exp(a.q*dt)
        st.latex(r'He^{q\Delta t}S_u-RB=V_u,\qquad He^{q\Delta t}S_d-RB=V_d')
        st.dataframe(pd.DataFrame({'Component':['Stock price ($)','Shares after reinvestment','Stock position ($)','Financing cash flow ($)','Replicated payoff ($)','Option next value ($)'],
            'Up':[S*t['u'],terminal_shares,terminal_shares*S*t['u'],-R*B,terminal_shares*S*t['u']-R*B,vU],
            'Down':[S*t['d'],terminal_shares,terminal_shares*S*t['d'],-R*B,terminal_shares*S*t['d']-R*B,vD]}).round(6),hide_index=True)


def parity(a):
    c,p=price(a,'Call'),price(a,'Put')
    rhs=a.S*exp(-a.q*a.T)-a.K*exp(-a.r*a.T)
    st.subheader('European put-call parity · equivalent terminal cash flows')
    st.latex(r'C-P=Se^{-qT}-Ke^{-rT}\quad\Longleftrightarrow\quad C+Ke^{-rT}=P+Se^{-qT}')
    st.latex(rf'{c:.6f}-{p:.6f}={a.S:g}e^{{-{a.q:g}({a.T:g})}}-{a.K:g}e^{{-{a.r:g}({a.T:g})}}={rhs:.6f}')
    st.write(f'Numerical difference: {c-p-rhs:.2e}. A call plus a riskless deposit matches a put plus a prepaid stock position. The stock position starts with exp(−qT) shares and grows to one share through dividend reinvestment.')
    st.caption('European parity is an equality for matching strike and maturity. Do not apply this equality to American options.')


def replication_first(a,t,kind):
    """Find financing from the down-state cash flow before valuing the claim."""
    dt,u,d,R=t['dt'],t['u'],t['d'],t['R']
    vd,vu=map(float,t['values'][1]);S=a.S
    H=exp(-a.q*dt)*(vu-vd)/(S*(u-d))
    shares=H*exp(a.q*dt)
    B=(shares*S*d-vd)/R
    V=H*S-B
    st.markdown('**First reproduce the payoff**')
    st.latex(r'He^{qT}uS-RB=V_u,\qquad He^{qT}dS-RB=V_d')
    st.write('Subtract the equations to find H. Then use the down-state equation to find the financing. Neither calculation needs an actual probability.')
    st.latex(r'H=e^{-qT}\frac{V_u-V_d}{S(u-d)}')
    st.latex(rf'H=e^{{-{a.q:.4f}({dt:.4f})}}\frac{{{vu-vd:.4f}}}{{{S*(u-d):.4f}}}={H:.6f}')
    st.latex(r'B=\frac{He^{qT}dS-V_d}{R}')
    st.latex(rf'B=\frac{{{shares*S*d:.4f}-{vd:.4f}}}{{{R:.6f}}}={B:.6f}')
    st.latex(rf'V=HS-B={H*S:.4f}-({B:.4f})={V:.6f}')
    st.write(('Buy ' if H>=0 else 'Short ')+f'**{abs(H):.6f} shares**; '+('borrow ' if B>=0 else 'lend ')+f'**${abs(B):.6f}** today.')
    st.dataframe(pd.DataFrame({'Component':['Stock price ($)','Shares after reinvestment','Stock position ($)','Financing cash flow ($)','Replicated payoff ($)','Option payoff ($)'],
        'Up':[S*u,shares,shares*S*u,-R*B,shares*S*u-R*B,vu],
        'Down':[S*d,shares,shares*S*d,-R*B,shares*S*d-R*B,vd]}).round(6),hide_index=True)
    st.markdown('**Now reconcile with risk-neutral valuation**')


def calculator_panel(a,prefix,extras=None):
    dt,u,d,R,p=a.terms()
    symbols=dict(S=a.S,K=a.K,T=a.T,r=a.r,q=a.q,sigma=a.sigma,n=a.n,dt=dt,u=u,d=d,R=R,p=p)
    symbols.update(extras or {})
    with st.expander('Symbol calculator · work with the current inputs'):
        st.write('Use +, -, *, /, parentheses and ^ for powers; exp, log, sqrt, abs, min, max are available. A leading = is optional. Rates are decimals here. Standard precedence applies: -2^2 = -4; (-2)^2 = 4.')
        st.dataframe(pd.DataFrame({'Symbol':list(symbols),'Current value':list(symbols.values())}),hide_index=True)
        expression=st.text_input('My expression',value='(exp((r-q)*dt)-d)/(u-d)',key=prefix+'_expression')
        cols=st.columns(3)
        if cols[0].button('Calculate',key=prefix+'_calc'):
            try:
                result=calculate(expression,symbols)
                st.session_state[prefix+'_result']=(expression,json.dumps(symbols,sort_keys=True),result)
            except ValueError as e: st.error(str(e))
        if cols[1].button('Clear calculation',key=prefix+'_clear'):
            st.session_state.pop(prefix+'_result',None)
        saved=st.session_state.get(prefix+'_result')
        valid=saved is not None and saved[0]==expression and saved[1]==json.dumps(symbols,sort_keys=True)
        if valid:
            st.code(expression,language=None)
            substituted=expression
            import re
            substituted=re.sub(r'\b[A-Za-z_]\w*\b',lambda m:f'({symbols[m[0]]:.8g})' if m[0] in symbols else m[0],expression)
            st.write('Substitution: '+substituted)
            st.success(f'Result = {saved[2]:.10g}')
            st.caption('Units follow your expression: option values are dollars/share, holdings are shares, rates/probabilities are dimensionless. No automatic dimensional analysis is performed.')
            if prefix+'_attempt' in st.session_state:
                cols[2].button('Use result as my answer',key=prefix+'_transfer',on_click=lambda:st.session_state.update({prefix+'_attempt':max(0.,saved[2])}))
        elif saved:
            st.info('Inputs or expression changed. Calculate again to refresh this result.')


def exports(a,trees,prefix):
    with st.expander('Export my calculations'):
        df=pd.concat([node_table(a,t).assign(Contract=label) for label,t in trees.items()],ignore_index=True)
        st.download_button('Download tree values (CSV)',df.to_csv(index=False).encode(),f'{prefix}_tree.csv','text/csv',key=prefix+'_csv')
        record={'inputs':a.__dict__,'contracts':{label:{'kind':t['kind'],'style':t['style'],'exercise_steps':sorted(t['allowed']),'barrier':t['barrier']} for label,t in trees.items()}}
        st.download_button('Download input record (JSON)',json.dumps(record,indent=2),f'{prefix}_inputs.json','application/json',key=prefix+'_json')


def core(n,prefix):
    st.title(ROUTES[prefix])
    st.write('**Goal:** '+('Reproduce call and put payoffs using stock and financing; reconcile replication with pricing probabilities.' if n==1 else 'Value cash flows backward and explain when the holder should exercise.'))
    presets=dict(K=100.,r=100*log(1.1)) if n==1 else dict(K=90.,r=10.,sigma=40.) if n==2 else {}
    a=inputs(prefix,n,manual=n==1,presets=presets)
    guided(a,prefix)
    reveal=st.toggle('Show worked calculations and trees',value=False,key=prefix+'_reveal')
    if reveal:
        st.latex(r'C_T=\max(S_T-K,0),\qquad P_T=\max(K-S_T,0)')
        styles=['European'] if n==1 else st.multiselect('Compare exercise styles',['European','American'],default=['European','American'],key=prefix+'_styles')
        upto=st.slider('Reveal backward from expiry: earliest visible calculation step',0,n,0,key=prefix+'_step') if n>1 else 0
        trees={}
        for style in styles:
            st.subheader(style+' options')
            cols=st.columns(2)
            for col,kind in zip(cols,['Call','Put']):
                t=lattice(a,kind,style);trees[style+' '+kind]=t
                with col:
                    st.metric(style+' '+kind+' · value today',f'${t["values"][0][0]:.6f}')
                    display={**t,'values':[v.copy() for v in t['values']]}
                    if upto>0:
                        # Hide earlier answers while retaining all prices/payoffs in the explanatory table.
                        for k in range(upto): display['values'][k][:]=np.nan
                    st.plotly_chart(tree_plot(a,display,style+' '+kind),width='stretch',config={'displayModeBar':False},key=prefix+'_'+style+'_'+kind)
                    if n==1:
                        replication_first(a,t,kind)
                        explain_node(a,t,0,0,kind,style,False)
        if upto>0: st.info('A question mark hides an earlier option value. Move the reveal control toward zero to complete backward induction. Full calculations remain available in the node walkthrough on request.')
        if n>1 and trees:
            st.subheader('Inspect a node · formula, substitution, and decision')
            cols=st.columns(3)
            selected=cols[0].selectbox('Contract to inspect',list(trees),key=prefix+'_contract')
            i=cols[1].selectbox('Time step before expiry',list(range(n)),key=prefix+'_i')
            j=cols[2].selectbox('Number of up moves',list(range(i+1)),key=prefix+'_j')
            t=trees[selected];style,kind=selected.split()
            explain_node(a,t,i,j,kind,style,st.checkbox('Also show replication at this node',key=prefix+'_hedge'))
        parity(a)
        if trees: exports(a,trees,prefix)
    calculator_panel(a,prefix)
    st.info('What we learned: price comes from matching cash flows, while exercise compares cash now with remaining flexibility. These trees assume frictionless trading, constant rates/yield, and specified price moves.')


def probability_page():
    st.title('Probabilities, leverage, and expected returns')
    st.write('**Optional goal:** distinguish valuation from forecasting and explain why a call’s expected return changes as its leverage changes. This page uses European calls; no dividends is the starting example.')
    a=inputs('probabilities',8,presets=dict(r=8.,sigma=30.,q=0.))
    mu=st.slider('Assumed actual stock-price growth mu (%/year)',-30.,60.,12.,.5,key='probabilities_mu')/100
    st.caption('mu is defined by E_actual[S_next]/S = exp(mu Δt). It is the log of the expected growth factor per year, not the expected log return. With dividends reinvested, expected total gross stock return is exp((mu+q)Δt).')
    try:pa=actual_probability(a,mu)
    except ValueError as e:st.error(str(e));return
    t=lattice(a);dt,u,d,R,p=a.terms()
    st.latex(r'p_{\mathrm{actual}}=\frac{e^{\mu\Delta t}-d}{u-d},\qquad p=\frac{e^{(r-q)\Delta t}-d}{u-d}')
    st.latex(rf'p_{{\mathrm{{actual}}}}={pa:.6f},\qquad p={p:.6f}')
    st.write('The probabilities below are model assumptions and pricing weights, respectively. The pricing model does not estimate the actual distribution. Holding today’s stock price and tree inputs fixed isolates the distinction.')
    stock=t['stock'][-1];physical=distribution(a.n,pa);neutral=distribution(a.n,p)
    cols=st.columns(2)
    for col,weights,title,color in [(cols[0],physical,'Assumed actual distribution','#007F86'),(cols[1],neutral,'Risk-neutral pricing distribution','#0756A3')]:
        fig=go.Figure(go.Bar(x=[f'{s:.2f}' for s in stock],y=weights,marker_color=color))
        fig.update_layout(title=title);fig.update_xaxes(title='Terminal stock price ($/share)');fig.update_yaxes(title='Probability',range=[0,1],tickformat='.0%')
        col.plotly_chart(plot_style(fig),width='stretch',config={'displayModeBar':False})
    st.latex(r'\Pr(J=j)=\binom{n}{j}p^j(1-p)^{n-j},\qquad S_T=S_0u^jd^{n-j}')
    C=float(t['values'][0][0]);expected=float(physical@t['payoff'][-1]);rn=float(neutral@t['payoff'][-1])/exp(a.r*a.T)
    m=st.columns(3);m[0].metric('No-arbitrage call value ($/share)',f'{C:.6f}');m[1].metric('Actual expected terminal payoff ($/share)',f'{expected:.6f}');m[2].metric('Actual expected holding-period return',f'{expected/C-1:.2%}' if C>1e-10 else 'Undefined')
    st.latex(r'C_0=e^{-rT}E_{RN}[C_T],\qquad E_{actual}[\text{holding return}]=E_{actual}[C_T]/C_0-1')
    st.caption(f'Independent terminal-distribution price check: ${rn:.6f}. The holding return is for buying and holding this European call to expiry; no annualization is applied to this metric.')
    st.subheader('Derivation · the option is a financed stock exposure')
    st.write('At any pre-expiry node, match the next-step option values using H shares with reinvested dividends and B dollars borrowed.')
    st.latex(r'C=HS-B,\qquad C_{next}=He^{q\Delta t}S_{next}-RB')
    st.write('Take an expectation using the actual probability. Define G_S as the expected gross total return on the dividend-reinvested stock position.')
    st.latex(r'G_S=e^{q\Delta t}\frac{E_{actual}[S_{next}]}{S}=e^{(\mu+q)\Delta t}')
    st.latex(r'E_{actual}[C_{next}]=HS\,G_S-RB=RC+HS(G_S-R)')
    st.latex(r'\boxed{G_C=\frac{E_{actual}[C_{next}]}{C}=R+\underbrace{\frac{HS}{C}}_{\text{stock exposure per dollar}}(G_S-R)}')
    st.write('This exact one-step identity explains expected returns. Calling them required returns additionally assumes equilibrium consistency between prices, beliefs, and risk compensation. An assumed actual probability alone does not establish that equilibrium.')
    st.latex(r'\rho_C=G_C-1,\qquad \mu_C^{growth}=\frac{\log G_C}{\Delta t}')
    st.caption('The annualized log expected-growth rate is log(E[C_next]/C)/Δt. It is not E[log(C_next/C)]/Δt, and it is not the realized annual return.')
    cols=st.columns(2);i=cols[0].selectbox('Inspect time step',list(range(8)),key='probabilities_i');j=cols[1].selectbox('Inspect up moves',list(range(i+1)),key='probabilities_j')
    result=node_return(a,t,i,j,pa)
    explain_node(a,t,i,j,'Call','European',True)
    if result is None:st.warning('Call value is zero at this node. A return per dollar invested is undefined.')
    else:
        st.latex(rf'HS/C={result["leverage"]:.6f},\quad G_C={R:.6f}+{result["leverage"]:.6f}({result["stock_gross"]:.6f}-{R:.6f})={result["gross"]:.6f}')
        st.write(f'Expected return over this {dt:.6f}-year step: **{result["simple"]:.2%}**. Expected stock total return over the same step: **{result["stock_gross"]-1:.2%}**.')
        st.write('Annualized log expected-growth rate: '+(f'{result["log_rate"]:.2%}' if result['log_rate'] is not None else 'undefined because expected payoff is zero')+'.')
    rows=[];labels={}
    for k in range(8):
        for j0 in range(k+1):
            z=node_return(a,t,k,j0,pa)
            labels[k,j0]='undefined' if z is None else f'{z["simple"]:.2%}'
            rows.append({'Step':k,'Up moves':j0,'Stock ($)':t['stock'][k][j0],'Call ($)':t['values'][k][j0],
                         'H':None if z is None else z['H'],'HS/C':None if z is None else z['leverage'],
                         'Expected step return':None if z is None else z['simple'],
                         'Annual log expected-growth rate':None if z is None else z['log_rate']})
    with st.expander('Full eight-period lattice and expected-return table'):
        st.plotly_chart(tree_plot(a,t,'European call · hover over a node for its expected step return',return_labels=labels),width='stretch',config={'displayModeBar':False})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.download_button('Download node returns (CSV)',pd.DataFrame(rows).to_csv(index=False),'eight_period_returns.csv','text/csv')
    st.subheader('Why deeper in the money can mean a lower expected return')
    st.write('With no dividends, a deeply in-the-money call approaches a stock position financed partly by borrowing. H approaches one, but the call remains leveraged until the discounted strike is small relative to S. Then HS/C approaches one and its expected return approaches the stock’s expected return. The direction of the excess-return effect depends on the assumed stock risk premium.')
    st.latex(r'q=0:\quad C\approx S-Ke^{-r\tau},\quad H\approx1,\quad \frac{HS}{C}\approx\frac{S}{S-Ke^{-r\tau}}\longrightarrow1')
    moneyness=[]
    for s in np.linspace(max(1,a.K*.5),max(200,a.K*5),35):
        b=replace(a,S=float(s));bt=lattice(b);z=node_return(b,bt,0,0,pa)
        if z:moneyness.append((s/b.K if b.K>0 else s,z['simple'],z['stock_gross']-1,z['leverage']))
    if moneyness:
        arr=np.array(moneyness);fig=go.Figure()
        fig.add_scatter(x=arr[:,0],y=arr[:,1],name='Call expected step return',line=dict(color='#0756A3'))
        fig.add_scatter(x=arr[:,0],y=arr[:,2],name='Stock expected step total return',line=dict(color='#007F86',dash='dash'))
        fig.update_xaxes(title='S/K (S in dollars if K=0)');fig.update_yaxes(title='Expected simple return per step',tickformat='.0%')
        st.plotly_chart(plot_style(fig),width='stretch',config={'displayModeBar':False})
    st.info('Try mu = r−q: actual and pricing probabilities coincide, and every positive-value European call node has expected step return R−1. Different actual probabilities change expected returns while leaving the model price fixed.')
    calculator_panel(a,'probabilities',dict(pa=pa,mu=mu,C=C))


@st.cache_data(show_spinner=False)
def convergence_data(a,ns,kind):
    return pd.DataFrame([{'Periods':n,'European':price(replace(a,n=n),kind),'American':price(replace(a,n=n),kind,'American')} for n in ns])


def convergence_page():
    st.title('Convergence · how many steps are enough?')
    a=inputs('convergence',100)
    if a.u is not None:
        st.warning('Convergence uses volatility calibration so that changing n preserves the same annual volatility. Select Volatility calibration.');return
    cols=st.columns(2);max_n=cols[0].slider('Largest plotted step count',20,500,100,10,key='convergence_max');kind=cols[1].selectbox('Option type',['Call','Put'],key='convergence_kind')
    ns=tuple(sorted(set(list(range(1,min(max_n,20)+1))+list(range(25,max_n+1,5))+[max_n])))
    valid=tuple(n for n in ns if np.exp((a.r-a.q)*a.T/n)>np.exp(-a.sigma*np.sqrt(a.T/n)) and np.exp((a.r-a.q)*a.T/n)<np.exp(a.sigma*np.sqrt(a.T/n)))
    if len(valid)<len(ns):st.warning('Coarse step counts that violate no arbitrage are omitted.')
    df=convergence_data(a,valid,kind);bsm=black_scholes(a,kind)
    with st.spinner('Calculating a fine-tree comparison...'):
        reference=price(replace(a,n=1200),kind,'American')
        reference2=price(replace(a,n=1201),kind,'American')
    fig=go.Figure()
    for name,color in [('European','#0756A3'),('American','#007F86')]:fig.add_scatter(x=df['Periods'],y=df[name],name=name,mode='lines+markers',line=dict(color=color))
    fig.add_scatter(x=[min(valid),max(valid)],y=[bsm,bsm],mode='lines',line=dict(dash='dash',color='#142B49'),name='European Black–Scholes')
    fig.add_scatter(x=[min(valid),max(valid)],y=[reference,reference],mode='lines',line=dict(dash='dot',color='#B06A00'),name='American fine-tree estimate')
    fig.update_xaxes(title='Number of time steps n');fig.update_yaxes(title=kind+' value ($/share)')
    fig=plot_style(fig,550)
    fig.update_layout(legend=dict(orientation='h',y=1.22,x=0),margin=dict(t=110,b=60))
    st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
    st.write(f'European Black–Scholes: **{bsm:.6f} dollars/share**. American 1,200-step estimate: **{reference:.6f} dollars/share**; 1,201-step estimate: **{reference2:.6f} dollars/share**. Their difference is a stability diagnostic, not an error bound.')
    st.latex(r'\Delta t=T/n,\quad u_n=e^{\sigma\sqrt{T/n}},\quad d_n=1/u_n,\quad p_n=\frac{e^{(r-q)T/n}-d_n}{u_n-d_n}')
    st.write('Recalibrate the up/down factors and pricing probability for every n. Holding the factors fixed while adding steps would change the model’s annual volatility.')
    with st.expander('European benchmark equations'):
        st.latex(r'd_1=\frac{\log(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},\quad d_2=d_1-\sigma\sqrt T')
        st.latex(r'C=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2),\quad P=Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1)')
        st.write('N is the standard normal cumulative distribution function. At K=0 the limiting call value is S exp(−qT), and the put value is zero.')
    df['American minus European']=df.American-df.European;df['European minus Black-Scholes']=df.European-bsm
    st.dataframe(df,hide_index=True)
    st.download_button('Download convergence values (CSV)',df.to_csv(index=False),'convergence.csv','text/csv')
    st.info('What we learned: prices can oscillate with odd and even step counts. Inspect several refinements. A finer numerical answer does not remove model assumptions.')


def bermudan_page():
    st.title('Bermudan options · what is flexibility worth?')
    st.write('**Goal:** separate permission to exercise from the decision to exercise. A holder can ignore additional opportunities, so more permitted dates cannot reduce the value on the same tree.')
    n=st.slider('Number of tree periods',2,12,6,key='bermudan_n')
    a=inputs('bermudan',n)
    kind=st.selectbox('Option type',['Put','Call'],key='bermudan_kind')
    dates=st.multiselect('Permitted early exercise steps (expiry is always included)',list(range(n)),default=[n//2],format_func=lambda i:f'Step {i}: {i*a.T/n:.4f} years',key='bermudan_dates')
    st.latex(r'V_i=\begin{cases}\max(V_i^{GO},V_i^{STOP}),&i\in\mathcal E\\V_i^{GO},&i\notin\mathcal E\end{cases}')
    st.write('Exercise-date set E = {'+', '.join(str(i) for i in sorted(set(dates)|{n}))+'}. Time 0 is included only if you explicitly select it.')
    fig=go.Figure(go.Scatter(x=[i*a.T/n for i in range(n+1)],y=[0]*(n+1),mode='markers+text',
        text=['Exercise' if i in set(dates)|{n} else 'Hold' for i in range(n+1)],textposition=['top center' if i%2==0 else 'bottom center' for i in range(n+1)],
        marker=dict(size=13,symbol=['diamond' if i in set(dates)|{n} else 'circle' for i in range(n+1)],color=['#007F86' if i in set(dates)|{n} else '#0756A3' for i in range(n+1)])))
    fig.update_xaxes(title='Time (years)',range=[-.12*a.T,1.12*a.T]);fig.update_yaxes(visible=False,range=[-1,1]);st.plotly_chart(plot_style(fig,230),width='stretch',config={'displayModeBar':False})
    trees={style:lattice(a,kind,style,dates) for style in ['European','Bermudan','American']}
    cols=st.columns(3)
    for col,(style,t) in zip(cols,trees.items()):col.metric(style+' '+kind,f'${t["values"][0][0]:.4f}')
    st.caption('Values in dollars/share: '+'; '.join(f'{style} {t["values"][0][0]:.6f}' for style,t in trees.items())+'.')
    st.latex(r'V_{European}\leq V_{Bermudan}\leq V_{American}')
    t=trees['Bermudan'];st.plotly_chart(tree_plot(a,t,'Bermudan '+kind),width='stretch',config={'displayModeBar':False})
    st.caption('Diamond: strict exercise; square: indifference. The date timeline shows permission even where exercise is not optimal.')
    cols=st.columns(2);i=cols[0].selectbox('Inspect time step',list(range(n)),key='bermudan_i');j=cols[1].selectbox('Inspect up moves',list(range(i+1)),key='bermudan_j')
    explain_node(a,t,i,j,kind,'Bermudan')
    exports(a,trees,'bermudan')
    st.info('Try removing every early date: the Bermudan becomes European. Select all dates: it becomes American on this tree. Compare adding dates while keeping n fixed; changing n also changes the stock-price approximation.')


def barrier_page():
    st.title('Protection that switches off · an up-and-out call')
    st.write('**Optional goal:** explain why cheaper protection can leave an exposure. This European call pays no rebate and terminates if S reaches or exceeds barrier L on a monitoring date. Monitoring occurs at time 0 and every tree date, including expiry; it is not continuous monitoring.')
    n=st.slider('Number of monitored periods',2,12,4,key='barrier_n');a=inputs('barrier',n)
    L=st.number_input('Upper barrier L ($/share)',.01,100000.,140.,step=5.,key='barrier_L')
    t=lattice(a,'Call','European',barrier=L);plain=price(a);out=float(t['values'][0][0])
    st.latex(r'C_T=\max(S_T-K,0)\,\mathbf1\{S_{t_i}<L\text{ at every monitoring date}\}')
    st.latex(r'C_i=\begin{cases}0,&S_i\geq L\\e^{-r\Delta t}[pC_u+(1-p)C_d],&S_i<L\end{cases}')
    cols=st.columns(3);cols[0].metric('Ordinary European call',f'${plain:.6f}');cols[1].metric('Up-and-out call',f'${out:.6f}');cols[2].metric('Price saving',f'${plain-out:.6f}')
    if a.S>=L:st.warning('The initial stock price already reaches the barrier, so the contract is knocked out at time 0.')
    st.plotly_chart(tree_plot(a,t,'Values conditional on still being alive at arrival'),width='stretch',config={'displayModeBar':False})
    st.caption('After a barrier hit the option stays dead. Values at later below-barrier nodes apply only to paths that have survived; a stock-price node alone does not record past knockout history.')
    cols=st.columns(2)
    i=cols[0].selectbox('Inspect barrier time step',list(range(n)),key='barrier_i')
    j=cols[1].selectbox('Inspect barrier up moves',list(range(i+1)),key='barrier_j')
    if t['decisions'][i][j]=='Knocked out':
        st.write(f'At step {i}, stock price {t["stock"][i][j]:.6f} dollars reaches barrier {L:.6f} dollars: option value is zero, with no rebate.')
    else:
        explain_node(a,t,i,j,'Call','European')
    st.subheader('Follow a path · the same terminal price can hide different histories')
    st.caption('With the guided four-period inputs, compare UUUD with DUUU: the same final stock price can have different barrier histories.')
    path=st.text_input('Path: U for up, D for down',value='U'*(n-1)+'D',max_chars=12,key='barrier_path').upper().strip()
    if len(path)!=n or any(c not in 'UD' for c in path):st.info(f'Enter exactly {n} letters, using U and D.')
    else:
        s=a.S;alive=s<L;rows=[{'Step':0,'Stock ($)':s,'Contract':'Alive' if alive else 'Knocked out'}]
        for i,c in enumerate(path,1):
            s*=t['u'] if c=='U' else t['d'];alive=alive and s<L*(1-1e-12)
            rows.append({'Step':i,'Stock ($)':s,'Contract':'Alive' if alive else 'Knocked out'})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.write(f'Ordinary terminal call payoff: **{max(s-a.K,0):.6f} dollars/share**. Barrier terminal payoff: **{max(s-a.K,0) if alive else 0:.6f} dollars/share**.')
    st.write('Try the same number of U and D moves in a different order. A path that hits the barrier can lose the protection precisely when prices rise strongly. The lower premium comes from a weaker contractual promise.')
    exports(a,{'Up-and-out call':t},'barrier')
    st.info('Monitoring frequency is part of the contract. A tree that monitors only at its dates does not automatically price a continuously monitored barrier. This page is contract exploration, not a convergence benchmark.')


def practice_page():
    st.title('Practice · explain the economics')
    st.write('These questions are ungraded. Hints and explanations are available on request; retry as often as you like.')
    topic=st.selectbox('Topic',['All']+list(dict.fromkeys(q[0] for q in QUESTIONS)),key='practice_topic')
    indices=[i for i,q in enumerate(QUESTIONS) if topic=='All' or q[0]==topic]
    idx=st.selectbox('Question',indices,format_func=lambda i:QUESTIONS[i][1],key='practice_question')
    tag,prompt,options,correct,explanations=QUESTIONS[idx]
    st.subheader(prompt)
    answer=st.radio('My answer',range(len(options)),format_func=options.__getitem__,index=None,key=f'practice_answer_{idx}')
    cols=st.columns(3)
    if cols[0].button('Check answer'):
        if answer is None:st.info('Choose an answer first, or reveal the explanation.')
        elif answer==correct:st.success(explanations[answer])
        else:st.warning(explanations[answer])
    if cols[1].button('Retry',on_click=lambda:st.session_state.update({f'practice_answer_{idx}':None})):pass
    with st.expander('Hint'):
        st.write({'Replication':'Match cash flows state by state.','Dividends':'Separate price appreciation, dividend income, and discounting.','Exercise':'What do you give up by exercising?','Returns':'Separate shares held from exposure per dollar invested.'}.get(tag,'Separate the contract’s cash flows, the model’s assumptions, and the decision being made.'))
    with st.expander('Reveal answer and explain all choices'):
        st.write('**Answer: '+options[correct]+'**')
        for opt,explanation in zip(options,explanations):st.write('**'+opt+'** — '+explanation)


def home():
    st.title('The Binomial Option Learning Studio')
    st.write('Price a promise by understanding its cash flows, reproducing its risk, and deciding when to exercise.')
    st.info('Start with a small tree you can explain. Then explore more dates, different exercise rights, and the distinction between a fair price and an expected return.')
    st.subheader('Your route through the studio')
    descriptions={
      'one':'Core · Calls left, puts right. Match both state payoffs with H shares and B dollars borrowed, with or without dividends. Verify European parity.',
      'two':'Core · Reveal backward induction and compare European/American values. Select a node to see formula and substitution.',
      'three':'Core · Compare calls and puts across exercise styles; investigate dividend and interest-rate effects.',
      'probabilities':'Optional · Compare eight-period terminal distributions and derive the leverage/expected-return relationship.',
      'convergence':'Core · Investigate numerical stability for European and American calls or puts.',
      'bermudan':'Extension · Choose finite exercise dates and measure the value of flexibility.',
      'barrier':'Optional · Investigate path dependence and why protection can cost less when it can switch off.',
      'practice':'Ungraded · Check understanding, obtain explanations, and retry.'}
    for page,description in descriptions.items():
        st.markdown(f'**[{ROUTES[page]}](?page={page})**  \n{description}')
    st.subheader('How to use this for homework')
    st.write('Enter the question’s exact inputs. Make a prediction before revealing a solution. Select nodes to show your formulas and substitutions; export your inputs and tree values if useful. Explain what the result means rather than submitting only a price. The examples here are practice examples, not answers to a graded assignment.')
    st.latex(r'C_T=\max(S_T-K,0),\quad P_T=\max(K-S_T,0),\quad V=e^{-r\Delta t}[pV_u+(1-p)V_d]')
    st.write('Prerequisites: option payoffs, borrowing/lending, present value, and the law of one price. Inputs and outputs are per share; multiply dollar amounts and share holdings by the contract quantity when a question specifies one.')
    with st.expander('Notation, model assumptions, and privacy'):
        st.markdown('**S** stock price; **K** strike; **T** maturity in years; **r** annual continuously compounded riskless rate; **q** continuous dividend yield; **sigma** annualized volatility; **H** shares held; **B** dollars borrowed; **p** pricing up probability. Positive B means borrowing; negative B means lending.')
        st.write('Frictionless trading; identical borrowing/lending rates; specified stock moves; constant inputs within a tree; dividend reinvestment for replication. Payoff excludes the initial premium. Value is the price today. A return divides subsequent value by the initial investment. Zero-value positions do not have a defined percentage return.')
        st.write('No login, grades, student names, or external data feeds. Session entries are temporary; download records if you need to retain them. Public hosting can maintain its own infrastructure logs.')


if route=='home':home()
elif route in ('one','two','three'):core({'one':1,'two':2,'three':3}[route],route)
elif route=='probabilities':probability_page()
elif route=='convergence':convergence_page()
elif route=='bermudan':bermudan_page()
elif route=='barrier':barrier_page()
elif route=='practice':practice_page()
