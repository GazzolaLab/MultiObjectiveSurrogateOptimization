: sAHP conductance

NEURON {
	SUFFIX sAHP
	USEION ca READ cai
	USEION k READ ek WRITE ik
	RANGE gbar, g, i
	GLOBAL tau, taucadiv, n, kca, cah
	GLOBAL tau1Ref, tau2, oinf, c1inf, CaRef
	RANGE a2, a1max, b1, b2, tau1
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
	(pS) = (picosiemens)
	(um) = (micron)
}

PARAMETER {
	gbar = .0  		(pS/um2)	
	tau = 9		    (ms)
	taucadiv = 1
	n = 3
	tau1Ref = 400 (ms)
	tau2 = 200    (ms)
	c1inf = 0.25
	oinf = 0.5
	CaRef = .002  (mM)
	kca = 0.001   (mM)
	cah = 0.01    (mM)
}

ASSIGNED {
	v			(mV)
	ek		(mV)
	ik		(mA/cm2)
	i 		(mA/cm2)
	ica		(mA/cm2)
        cai             (mM)
  g     (pS/um2)
  a1ca  (/ms)
  a1maxCaRef  (/ms)
  a1max (/ms/mM)
  a1    (/ms/mM)
  b1    (/ms)
  a2    (/ms)
  b2    (/ms)
  tau1  (ms)
}

STATE { 
	c1 
	o 
}

INITIAL {
	a2 = -(oinf/((-1 + c1inf)*tau2))
	b1 = -(c1inf/((-1 + oinf)*tau1Ref))
	b2 = 1/tau2 - a2
	a1maxCaRef = 1/tau1Ref - b1 
	a1max = a1maxCaRef/CaRef
	a1ca = (a1max*cai)/(1+exptrap(1,-(cai-cah)/kca))
	tau1 = 1/(a1ca + b1)
	o = (a2*(-1 + b1*tau1)*tau2)/(-1 + a2*b1*tau1*tau2)
	c1 = (b1*tau1*(-1 + a2*tau2))/(-1 + a2*b1*tau1*tau2)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = gbar*o^n
	ik = g*(v - ek)*(1e-4)
	i = ik
}

DERIVATIVE state {
	a1ca = (a1max*cai)/(1+exptrap(2,-(cai-cah)/kca))
	c1' = b1*(1 - o) - c1*(a1ca + b1)
	o' = a2*(1 - c1) - o*(a2 + b2)
}

FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}
