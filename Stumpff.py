# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:33:20 2018

@author: Haider
"""
import numpy as n


def S(z):

    if z == 0:

        S = 1/6

    elif z > 0:

        S = (n.sqrt(z) - n.sin(n.sqrt(z)))/n.sqrt((z**3))

    elif z < 0:

        S = n.sinh(n.sqrt(-z)) - (n.sqrt(-z))/(n.sqrt(-z))**3

    return S


def C(z):

    if z == 0:

        C = 0.5

    elif z > 0:

        C = (1 - n.cos(n.sqrt(z)))/z

    elif z < 0:

        C = (n.cosh(n.sqrt(-z)) - 1)/-z

    return C
