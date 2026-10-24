import sys
sid = sys.argv[1]
txt = {int(l.split('|')[0]): l.rstrip().split('|', 2)[2] for l in open('sessions/%s/phrases.txt' % sid, encoding='utf-8')}
todo = {int(l.split('|')[0]) for l in open('experiment/todo_%s.txt' % sid, encoding='utf-8')}
codes = {}
for l in open('experiment/codes_%s.txt' % sid, encoding='utf-8'):
    if l.startswith('#') or not l.strip():
        continue
    f = l.split()
    assert len(f) == 12, l
    codes[int(f[0])] = f[1:]
print('missing', sorted(todo - set(codes)), 'extra', sorted(set(codes) - todo))
lo, hi = int(sys.argv[2]), int(sys.argv[3])
for i in sorted(codes):
    if lo <= i <= hi:
        c = codes[i]
        print('%d %s/%s | %s' % (i, c[0], c[1], txt.get(i, '?')[:40]))
