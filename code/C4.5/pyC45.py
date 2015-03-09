import math
from  xml.dom import  minidom
from xml.etree import ElementTree as ET

# entropy 구하는 함수 
def getEntropy(s):
  entropy = 0
  for i in set(s):
    p_i = float(s.count(i)) / len(s)
    entropy = entropy - p_i * math.log(p_i, 2)
  return entropy

def train(training_obs, training_cat, xmldir):
  # 학습용 data 객체와 해당하는 카테고리 깂의 개수가 다르면 학습실패
  if not len(training_obs) == len(training_cat):
    return False

  # 속성값 추가
  attrs_names = training_obs[0]
  data = [[] for i in range(len(attrs_names))]
  categories = []

  # 각 속성에 속하는 값으로 배열만들기 및 분류할 카테고리 만들기 
  for i in range(1,len(training_obs)):
    categories.append(training_cat[i])
    for j in range(len(attrs_names)):
      data[j].append(training_obs[i][j])

  root = ET.Element('DecisionTree')
  tree = ET.ElementTree(root)

  grow_tree(data, categories, root, attrs_names)
  
  tree.write(xmldir)
  ET.dump(prettify(root))
  
  return True

##########


def grow_tree(data, category, parent, attrs_names):
  # 같은 클래스에 있는지 확인 
  if len(set(category)) > 1:

    division = []

    # 속성값의 범위 만큼 반복 
    for i in range(len(data)):
      # 해당 속성값이 없는 데이터인지 확인
      if set(data[i]) == set("?"):
        division.append(0)
      else:
        # 해당 속성 값이 연속적인 값인지 확인
        if (isnum(data[i])):
          division.append(gain(category,data[i]))           
        else:
          division.append(gain_ratio(category,data[i]))

    if max(division) == 0:
      num_max = 0
      for cat in set(category):
        num_cat=category.count(cat)
        if num_cat>num_max:
          num_max=num_cat
          most_cat=cat                
      parent.text=most_cat
    else:
      index_selected=division.index(max(division))
      name_selected=attrs_names[index_selected]
      if isnum(data[index_selected]):
        div_point=division_point(category,data[index_selected])
        r_son_data=[[] for i in range(len(data))]
        r_son_category=[]
        l_son_data=[[] for i in range(len(data))]
        l_son_category=[]
        for i in range(len(category)):
          if not data[index_selected][i]=="?":
            if float(data[index_selected][i])<float(div_point):
              l_son_category.append(category[i])
              for j in range(len(data)):
                l_son_data[j].append(data[j][i])     
            else:
              r_son_category.append(category[i])
              for j in range(len(data)):
                r_son_data[j].append(data[j][i])  
        if len(l_son_category)>0 and len(r_son_category)>0:
          p_l=float(len(l_son_category))/(len(data[index_selected])-data[index_selected].count("?"))
          son=ET.SubElement(parent,name_selected,{'value':str(div_point),"flag":"l","p":str(round(p_l,3))})
          grow_tree(l_son_data,l_son_category,son,attrs_names)
          son=ET.SubElement(parent,name_selected,{'value':str(div_point),"flag":"r","p":str(round(1-p_l,3))})
          grow_tree(r_son_data,r_son_category,son,attrs_names)
        else:
          num_max=0
          for cat in set(category):
            num_cat=category.count(cat)
            if num_cat>num_max:
              num_max=num_cat
              most_cat=cat                
          parent.text=most_cat
      else:
          for k in set(data[index_selected]):
            if not k=="?":
              son_data=[[] for i in range(len(data))]
              son_category=[]
              for i in range(len(category)):
                if data[index_selected][i]==k:
                  son_category.append(category[i])
                  for j in range(len(data)):
                    son_data[j].append(data[j][i])
              son=ET.SubElement(parent,name_selected,{'value':k,"flag":"m",'p':str(round(float(len(son_category))/(len(data[index_selected])-data[index_selected].count("?")),3))}) 
              grow_tree(son_data,son_category,son,attrs_names)   
  else:
      parent.text = category[0]

def prettify(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            prettify(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

# 속성 값이 숫자인지 확인하는 함수
def isnum(attr):
  for x in set(attr):
    if not x == "?":
      try:
        x = float(x)
        return isinstance(x,float)
      except ValueError:
        return False
  return True
    
# 속성 값이 명목형 것일 때 정보이득을 구하는 함수 
def gain_ratio(category,attr):
  s = 0
  cat = []
  att = []

  # 알지못하는 속성값을 제외하기 
  for i in range(len(attr)):
    if not attr[i]=="?":
      cat.append(category[i])
      att.append(attr[i])

  # 정보이득 구하는 함수
  for i in set(att):      
    p_i = float(att.count(i)) / len(att)
    cat_i = []
    for j in range(len(cat)):
      if att[j] == i:
        cat_i.append(cat[j])
    s = s + p_i * getEntropy(cat_i)

  gain = getEntropy(cat) - s
  ent_att = getEntropy(att)

  if ent_att == 0:
    return 0
  else:
    return gain / ent_att

# 속성 값이 연속된 것일 때 정보이득을 구하는 함수 
def gain(category, attr):
  cats = []

  # 알지못하는 속성값을 제외하기 
  for i in range(len(attr)):
    if not attr[i]=="?":
      cats.append([float(attr[i]), category[i]])

  # 정렬 (오름차순)
  cats = sorted(cats, key=lambda x:x[0])
  
  # 분류 (카테고리, 속성값)
  cat = [cats[i][1] for i in range(len(cats))]
  att = [cats[i][0] for i in range(len(cats))]

  # 속성 값이 하나인지 확인 : 하나면은 같은 클래스에 존재.
  if len(set(att)) == 1:
    return 0
  else:
    gains=[]
    div_point=[]

    # 분할하고 각각의 E(A)를 구함
    # E(A): 하위 각 node의 entropy를 계산한 후 node의 속한 레코드의 개수를 가중치로 하여 엔트로피를 평균한 값
    # 단 속성 값이 변할때만 구하면 된다. 

    for i in range(1, len(cat)):
      if not att[i] == att[i-1]:
        gains.append((getEntropy(cat[:i]) * float(i) / len(cat)) + (getEntropy(cat[i:]) * (1 - float(i) / len(cat))))
        div_point.append(i)
    
    # 여기서 가장 작은 E(A)를 골라 정보 이득값을 구함
    gain = getEntropy(cat) - min(gains)
    index = gains.index(min(gains))

    p_1 = float(div_point[index]) / len(cat)

    # ??
    ent_attr = -p_1 * math.log(p_1, 2) - (1 - p_1) * math.log((1 - p_1), 2)
    return gain/ent_attr

def division_point(category,attr):
    cats=[]
    for i in range(len(attr)):
        if not attr[i]=="?":
            cats.append([float(attr[i]),category[i]])
    cats=sorted(cats, key=lambda x:x[0])
    
    cat=[cats[i][1] for i in range(len(cats))]
    att=[cats[i][0] for i in range(len(cats))]
    gains=[]
    div_point=[]
    for i in range(1,len(cat)):
        if not att[i]==att[i-1]:
            gains.append(getEntropy(cat[:i])*float(i)/len(cat)+getEntropy(cat[i:])*(1-float(i)/len(cat)))
            div_point.append(i)
    return att[div_point[gains.index(min(gains))]]

def add(d1,d2):
    d=d1
    for i in d2:
        if d.has_key(i):
            d[i]=d[i]+d2[i]
        else:
            d[i]=d2[i]
    return d

def decision(root,obs,attrs_names,p):
    if root.hasChildNodes():
        att_name=root.firstChild.nodeName
        if att_name=="#text":
            
            return decision(root.firstChild,obs,attrs_names,p)  
        else:
            att=obs[attrs_names.index(att_name)]
            if att=="?":
                d={}
                for child in root.childNodes:                    
                    d=add(d,decision(child,obs,attrs_names,p*float(child.getAttribute("p"))))
                return d
            else:
                for child in root.childNodes:
                    if child.getAttribute("flag")=="m" and child.getAttribute("value")==att or \
                        child.getAttribute("flag")=="l" and float(att)<float(child.getAttribute("value")) or \
                        child.getAttribute("flag")=="r" and float(att)>=float(child.getAttribute("value")):
                        return decision(child,obs,attrs_names,p)    
    else:
        return {root.nodeValue:p}

def predict(xmldir,testing_obs):
    doc = minidom.parse(xmldir)
    root=doc.childNodes[0]
    prediction=[]
    attrs_names=testing_obs[0]
    for i in range(1,len(testing_obs)):
        
        answerlist=decision(root,testing_obs[i],attrs_names,1)
        answerlist=sorted(answerlist.iteritems(), key=lambda x:x[1], reverse = True )
        answer=answerlist[0][0]
        prediction.append(answer)
    return prediction



    
