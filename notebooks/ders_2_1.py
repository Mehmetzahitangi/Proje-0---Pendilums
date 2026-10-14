import sympy as sp

# Durum, kuvvet ve parametreler: düz semboller (zamana bağlı fonksiyon değil, çünkü sadece cebir yapacağız)
x, xd, th, thd, F = sp.symbols('x xdot theta thetadot F')
M, m, l, g = sp.symbols('M m l g', positive=True)

# Aşama 1'in sonucu (cartpole.py'deki formüllerin aynısı)
den  = M + m*sp.sin(th)**2
xdd  = (F + m*sp.sin(th)*(l*thd**2 - g*sp.cos(th))) / den
thdd = (-F*sp.cos(th) - m*l*thd**2*sp.sin(th)*sp.cos(th) + (M + m)*g*sp.sin(th)) / (l*den)

s = sp.Matrix([x, xd, th, thd])        # durum vektörü
f = sp.Matrix([xd, xdd, thd, thdd])    # ṡ = f(s, F)

A = f.jacobian(s)                      # ∂f/∂s : 4×4 (her satır bir türev, her sütun bir durum)
B = f.jacobian([F])                    # ∂f/∂F : 4×1

denge = {x: 0, xd: 0, th: 0, thd: 0, F: 0}   # dik duruş, kuvvet yok
A0 = sp.simplify(A.subs(denge))
B0 = sp.simplify(B.subs(denge))
print("A =", A0)
print("B =", B0)

sayilar = {M: 1.0, m: 0.1, l: 0.5, g: 9.81}
print("A sayisal =", A0.subs(sayilar))
print("B sayisal =", B0.subs(sayilar))