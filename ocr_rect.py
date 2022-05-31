#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:55:02 2022

@author: ai
"""


import os
import re
from PIL import Image



def check_array(tex):
    side=tex
    az='abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    i=0
    yo=len(az)
    txt=''
    for i in range(0,yo):
        if side==az[i]:
            txt=az[i]
            break
    return txt   
def catch_rectify_plate_characters(text):
    tex = text
    out1=[]
    size=len(tex)
    for i in range(0,size):
      if tex[i]==check_array(tex[i]):
        out1.append(tex[i])
    yup=''.join(str(e) for e in out1)    
    return yup
def conv_smal_to_caps(text):
  small_alpa='abcdefghijklmnopqrstuvwxyz'
  capi_alpa='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  for i in range(len(text)):
    if text[i].isalpha():
      if text[i] in small_alpa:
        tip= ''.join(m for l,m in zip(small_alpa,capi_alpa) if text[i]==l)
        text = text.replace(text[i],tip)
  return text


def replacing_filteration(word):
  word = list(str(word))
  if len(word)==10:
    for i in range(len(word)):
      if i!=0 and i!=1 and i!=4 and i!=5: # letters to digits
        if word[i]=='L' or word[i]=='A' or word[i]=='V':
          word[i]='4'
        if word[i]=='Z':
          word[i]='7'
        if word[i]=='J' or word[i]=='I' or word[i]=='N':
          word[i] = '1'
        if word[i]=='U' or word[i]=='O' or word[i]=='C' or word[i]=='D':
          word[i] = '0'
        if word[i]=='E' or word[i]=='B' or word[i]=='K' or word[i]=='F' or word[i]=='H':
          word[i] = '8'
        if word[i]=='G':
           word[i]='6'
        if word[i]=='S':
           word[i]='5'
        if word[i]=='T':
          word[i] = '7'
        if word[i]=='R' or word[i]=='P':
          word[i] = '2'
      else: # digits to letters 
        if word[i]=='8'or word[i]=='3'or word[i]=='9':
          word[i] = 'B'
        if word[i]=='6':
          word[i] = 'G'
        if word[i]=='0':
          word[i] = 'O'
        if word[i]=='1':
          word[i] = 'I'
        if word[i]=='7':
          word[i] = 'T'
        if word[i]=='4':
          word[i] = 'A'
        if word[i]=='2':
          word[i] = 'P'
        if word[i]=='5':
          word[i] = 'S'
    word = ''.join(word)
  elif len(word)==9:
    for i in range(len(word)):
      if i!=0 and i!=1 and i!=4: # letters to digits
        if word[i]=='L' or word[i]=='A' or word[i]=='V':
          word[i]='4'
        if word[i]=='Z':
          word[i]='7'
        if word[i]=='J' or word[i]=='I' or word[i]=='N':
          word[i] = '1'
        if word[i]=='U' or word[i]=='O' or word[i]=='C':
          word[i] = '0'
        if word[i]=='E' or word[i]=='B' or word[i]=='K' or word[i]=='F':
          word[i] = '8'
        if word[i]=='G':
           word[i]='6'
        if word[i]=='S':
           word[i]='5'
        if word[i]=='T':
          word[i] = '7'
        if word[i]=='R' or word[i]=='P':
          word[i] = '2'
      else: # digits to letters 
        if word[i]=='8'or word[i]=='3'or word[i]=='9':
          word[i] = 'B'
        if word[i]=='0':
          word[i] = 'O'
        if word[i]=='6':
          word[i] = 'G'
        if word[i]=='1':
          word[i] = 'I'
        if word[i]=='7':
          word[i] = 'T'
        if word[i]=='4':
          word[i] = 'A'
        if word[i]=='2':
          word[i] = 'P'
        if word[i]=='5':
          word[i] = 'S'
    word = ''.join(word)
  return word

def get_text_from_image(txt):
    start_codes = ['AN', 'AP', 'AR', 'AS', 'BH', 'BR', 'CH', 'CG', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LA', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB','OR','UA','DN']
    final_text = ''
    if len(txt)!=0:
      if len(txt)>=2:
        # x = ' '.join(result[le][1] for le in range(len(result)))
        text = catch_rectify_plate_characters(txt)
      else:
        text = catch_rectify_plate_characters(txt)
      text=conv_smal_to_caps(text)
      res = bool(re.match("^(?=.*[a-zA-Z])(?=.*[\d])[a-zA-Z\d]+$", str(text)))
      # text=replacing_filteration(text)
      print('filtering:',text)
      final_text = text
      # if res==True and 11>len(str(text))>8 and str(text)[0].isalpha() and str(text)[1].isalpha() and text[:2] in start_codes:
      #   # print('It is Plate')
      

      
        
    return final_text



