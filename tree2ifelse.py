# -*- coding:utf-8 -*-
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import heapq
from sklearn.externals.six import StringIO
import jieba
import re
from sklearn.tree import DecisionTreeClassifier

def get_word(s, cut):
    return ' '.join(cut(str(s)))


def get_python(X, y, cut=jieba.cut, n=100,min_pro=0.75,func_name='function',max_depth=5, min_samples_leaf=50, max_leaf_nodes=20):
    '''
    
    :param X: 训练文本
    :param y: 训练标签
    :param cut: 分词器
    :param n: 返回关键词个数
    :param min_pro: 写入代码的最小概率
    :param func_name: 代码函数名称
    :param max_depth: 决策树参数->深度
    :param min_samples_leaf: 决策树参数->节点最少样本
    :param max_leaf_nodes: 决策树参数->最多叶子数
    :return:
    '''

    X = [get_word(i, cut) for i in X]
    vectorizer = CountVectorizer(max_features=3000)  # onehot的编码
    X = vectorizer.fit_transform(X).toarray()
    print('分词完毕，现在开始计算相关性...')

    PYTNON = '''def %s(s:str):
'''%(func_name)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
    clf.fit(X, y)
    print('训练完毕，现在开始寻找规则...')
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=vectorizer.get_feature_names(),
                         filled=True, rounded=True,
                         special_characters=True)

    tree_info = dot_data.getvalue()

    jd = []
    lj = []
    node_info = []
    for i in tree_info.split('\n'):
        if re.search('\d+ \[label=<.*?>', i):
            jd.append(re.search('(\d+) \[label=<(.*?)<', i).group(1, 2))
        if re.search('\d+.*?value = ', i):
            node_info.append(re.search('(\d+).*?value = (\[.*?\])', i).group(1, 2))


        if re.search('\d+ -> \d+', i):
            lj.append(re.search('(\d+) -> (\d+)', i).group(1, 2))


    root = dict(jd)
    node_info = dict(node_info)

    node = []
    dlj = []
    for i in lj:
        if int(i[0]) in node:
            dlj.append((int(i[0]), int(i[1]), True))
        else:
            dlj.append((int(i[0]), int(i[1]), False))
            node.append(int(i[0]))

    all_writed = []
    for i in X:
        if clf.apply([i])[0] in all_writed:
            continue
        else:
            all_writed.append(clf.apply([i])[0])
        pytnon = ''
        tab = '\t'
        if clf.predict_proba([i])[0][1] > min_pro:
            pytnon += '\t#节点信息%s\n' % (node_info[str(clf.apply([i])[0])])
            last_node = 0
            for inx, node in enumerate(clf.decision_path([i]).toarray()[0]):
                if node and inx:
                    for i in dlj:
                        if i[0] == last_node and i[1] == inx:
                            if i[2]:
                                pytnon += tab
                                s = root[str(i[0])].split(' &le; ')[0]
                                pytnon += "if '%s' in s:\n" % (s)
                                tab += '\t'
                            else:
                                pytnon += tab
                                s = root[str(i[0])].split(' &le; ')[0]
                                pytnon += "if '%s' not in s:\n" % (s)
                                tab += '\t'
                    last_node = inx

            pytnon += tab
            pytnon += "return 1\n"
            PYTNON += pytnon


    score = clf.feature_importances_
    a = score
    word = vectorizer.get_feature_names()
    x = heapq.nlargest(n, range(len(a)), a.take)
    res = []
    for w in x:
        res.append([word[w], a[w]])
    return PYTNON,res


if __name__ == "__main__":
    xs=[]
    ys=[]
    with open('train.txt',encoding='utf8') as f:
        for i in f:
            x,y = i.strip().split('\t')
            xs.append(x)
            ys.append(int(y))

    py,word = get_python(xs,ys)
    with open('1.py','w',encoding='utf8') as f:
        f.write(py)
