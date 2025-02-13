import pandas.io.sql as psql
import pandas as pd
import datetime
config = __import__('TPCHconfig-pandas')


mini01 = """
SELECT * FROM orders;
"""

mini02 = """
SELECT * FROM lineitem;
"""

mini03 = """
SELECT * FROM orders, lineitem WHERE o_order = l_orderkey AND o_orderpriority = '1-URGENT'
"""

def mini_05(db):
  """
  CREATE VIEW pt1 AS SELECT l_returnflag, l_linestatus,l_quantity, l_extendedprice, l_discount,l_extendedprice * (1 - l_discount) AS disc_price FROM lineitem where
    l_shipdate <= date '1995-12-2';

  SELECT count(*) FROM pt1 GROUP BY 
  l_returnflag, l_linestatus;
  """
  l = db.lineitem
  l = l[l['l_shipdate'] <= datetime.date(1995, 12, 2)]
  l['disc_price'] = l['l_extendedprice'] * (1 - l['l_discount'])
  l.materialize()
  l = l[['l_returnflag', 'l_linestatus', 'l_quantity', 'l_extendedprice', 'disc_price', 'l_discount']]
  l = l.groupby(('l_returnflag', 'l_linestatus'), sort=False)
  return l

def mini_06(db):
    p = db.part
    p = p[(p['p_size'] == 15) & (p['p_type'].str.contains('^.*BRASS$'))]
    p.materialize()
    p = p[['p_partkey', 'p_mfgr']]
    ps = db.partsupp
    ps = ps[['ps_suppkey', 'ps_supplycost', 'ps_partkey']]
    ps.materialize()
    s = db.supplier
    s.materialize()
    s = s[['s_nationkey', 's_suppkey', 's_acctbal', 's_name', 's_address', 's_phone', 's_comment']]
    n = db.nation
    n = n[['n_nationkey', 'n_regionkey', 'n_name']]
    r = db.region
    r = r[r['r_name'] == 'EUROPE']
    r = r[['r_regionkey']]
    r.materialize()

    j = ps.merge(s, left_on='ps_suppkey', right_on='s_suppkey')
    j = j.merge(n, left_on='s_nationkey', right_on='n_nationkey')
    j.materialize()
    j = j.merge(r, left_on='n_regionkey', right_on='r_regionkey')

    ti = j[['ps_partkey', 'ps_supplycost']].groupby('ps_partkey').min()


def q01(db):
    """
select
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
from
    lineitem
where
    l_shipdate <= date '1998-12-01' - interval '90' day (3)
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus;
    """
    l = db.lineitem
    l = l[l['l_shipdate'] <= datetime.date(1998, 9, 2)]
    l['disc_price'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l['charge'] = l['disc_price'] * (1 + l['l_tax'])
    l = l[['l_returnflag', 'l_linestatus', 'l_quantity', 'l_extendedprice', 'disc_price', 'charge', 'l_discount']]
    l = l.groupby(('l_returnflag', 'l_linestatus'), sort=False) \
        .agg({'l_quantity': ['sum', 'mean'], 'l_extendedprice': ['sum', 'mean'], 'disc_price': 'sum',
        'charge': 'sum', 'l_discount': ['mean', 'count']})
    l.columns = pd.Index([l.columns.levels[1][l.columns.labels[1][i]] + '_' + l.columns.levels[0][l.columns.labels[0][i]] for i in range(len(l.columns.labels[0]))])
    l.rename(columns={'sum_l_quantity': 'sum_qty', 'sum_l_extendedprice': 'sum_base_price',
        'mean_l_quantity': 'avg_qty', 'mean_l_extendedprice': 'avg_price',
        'mean_l_discount': 'avg_disc', 'count_l_discount': 'count_order'}, inplace=True)
    l.reset_index(inplace=True)
    l.sort_values(['l_returnflag', 'l_linestatus'], inplace=True)
    return l


def q02(db):
    """
select
	s_acctbal,
	s_name,
	n_name,
	p_partkey,
	p_mfgr,
	s_address,
	s_phone,
	s_comment
from
	part,
	supplier,
	partsupp,
	nation,
	region
where
	p_partkey = ps_partkey
	and s_suppkey = ps_suppkey
	and p_size = 15
	and p_type like '%BRASS'
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'EUROPE'
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp,
			supplier,
			nation,
			region
		where
			p_partkey = ps_partkey
			and s_suppkey = ps_suppkey
			and s_nationkey = n_nationkey
			and n_regionkey = r_regionkey
			and r_name = 'EUROPE'
	)
order by
	s_acctbal desc,
	n_name,
	s_name,
	p_partkey
limit 100;
-- rewritten to avoid corelated query
select
  s_acctbal,
  s_name,
  n_name,
  p_partkey,
  p_mfgr,
  s_address,
  s_phone,
  s_comment
from
  part,
  supplier,
  partsupp,
  nation,
  region,
	 (
    select
		  ps_partkey as i_partkey,
      min(ps_supplycost) as min_supplycost
    from
      partsupp,
      supplier,
      nation,
      region
    where
      s_suppkey = ps_suppkey
      and s_nationkey = n_nationkey
      and n_regionkey = r_regionkey
      and r_name = 'EUROPE'
		group by ps_partkey
  ) costtbl
where
  p_partkey = ps_partkey
  and s_suppkey = ps_suppkey
  and p_size = 15
  and p_type like '%BRASS'
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'EUROPE'
	and ps_partkey = i_partkey
  and ps_supplycost = min_supplycost
order by
  s_acctbal desc,
  n_name,
  s_name,
  p_partkey
limit 100;
    :param db:
    :return:
    """
    p = db.part
    p = p[(p['p_size'] == 15) & (p['p_type'].str.contains('^.*BRASS$'))]
    p = p[['p_partkey', 'p_mfgr']]
    ps = db.partsupp
    ps = ps[['ps_suppkey', 'ps_supplycost', 'ps_partkey']]
    s = db.supplier
    s = s[['s_nationkey', 's_suppkey', 's_acctbal', 's_name', 's_address', 's_phone', 's_comment']]
    n = db.nation
    n = n[['n_nationkey', 'n_regionkey', 'n_name']]
    r = db.region
    r = r[r['r_name'] == 'EUROPE']
    r = r[['r_regionkey']]

    j = ps.merge(s, left_on='ps_suppkey', right_on='s_suppkey')
    j = j.merge(n, left_on='s_nationkey', right_on='n_nationkey')
    j = j.merge(r, left_on='n_regionkey', right_on='r_regionkey')

    ti = j[['ps_partkey', 'ps_supplycost']].groupby('ps_partkey').min()
    ti.reset_index(inplace=True)
    ti.rename(columns={'ps_supplycost': 'min_supply_cost', 'ps_partkey': 'i_partkey'}, inplace=True)

    t = j.merge(p, left_on='ps_partkey', right_on='p_partkey')
    t = t.merge(ti, left_on=['ps_partkey', 'ps_supplycost'], right_on=['i_partkey', 'min_supply_cost'])
    t = t[['s_acctbal', 's_name', 'n_name', 'p_partkey', 'p_mfgr', 's_address', 's_phone', 's_comment']]
    t.sort_values(['s_acctbal', 'n_name', 's_name', 'p_partkey'], ascending=[False, True, True, True], inplace=True)
    return t.head(100)


def q03(db):
    """
 select
  l_orderkey,
  sum(l_extendedprice),
  o_orderdate,
  o_shippriority
from
  customer,
  orders,
  lineitem
where
  c_mktsegment = 'BUILDING'
  and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < date '1995-03-15'
  and l_shipdate > date '1995-03-15'
group by
  l_orderkey,
  o_orderdate,
  o_shippriority
order by
  revenue desc,
  o_orderdate
limit 10;
    """
    c = db.customer
    o = db.orders
    l = db.lineitem
    c = c[c['c_mktsegment'] == 'BUILDING']
    c = c[['c_custkey']]
    o = o[o['o_orderdate'] < datetime.date(1995, 3, 15)]
    o = o[['o_orderdate', 'o_shippriority', 'o_orderkey', 'o_custkey']]
    l = l[l['l_shipdate'] > datetime.date(1995, 3, 15)]
    l['revenue'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l = l[['l_orderkey', 'revenue']]

    t = c.merge(o, left_on='c_custkey', right_on='o_custkey')
    t = t.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t[['l_orderkey', 'revenue', 'o_orderdate', 'o_shippriority']]
    t = t.groupby(('l_orderkey', 'o_orderdate', 'o_shippriority'), sort=False).sum(min_count=1)
    t.reset_index(inplace=True)
    t.sort_values(['revenue', 'o_orderdate'], ascending=[False, True], inplace=True)
    t = t.head(10)
    return t


def q04(db):
    """
select
  o_orderpriority,
  count(*) as order_count
from
  orders
where
  o_orderdate >= date '1993-07-01'
  and o_orderdate < date '1993-07-01' + interval '3' month
  and exists (
    select
      *
    from
      lineitem
    where
      l_orderkey = o_orderkey
      and l_commitdate < l_receiptdate
  )
group by
  o_orderpriority
order by
  o_orderpriority;

-- rewritten to avoid corelated query

select
  o_orderpriority,
  count(*) as order_count
from
  orders
where
  o_orderdate >= date '1993-07-01'
  and o_orderdate < date '1993-07-01' + interval '3' month
  and o_orderkey IN (
    select
      l_orderkey
    from
      lineitem
    where
      l_commitdate < l_receiptdate
  )
group by
  o_orderpriority
order by
  o_orderpriority;
    """
    l = db.lineitem
    l = l[l['l_commitdate'] < l['l_receiptdate']]
    o = db.orders
    o = o[(o['o_orderdate'] >= datetime.date(1993, 7, 1))
          &(o['o_orderdate'] < datetime.date(1993, 10, 1))
          &(o['o_orderkey'].isin(l['l_orderkey']))]
    o = o[['o_orderpriority', 'o_orderkey']]

    t = o.groupby('o_orderpriority').count()
    t.rename(columns={'o_orderkey': 'order_count'}, inplace=True)
    t.reset_index(inplace=True)
    t.sort_values('o_orderpriority')
    return t


def q05(db):
    """
select
  n_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  customer,
  orders,
  lineitem,
  supplier,
  nation,
  region
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and o_orderdate >= date '1994-01-01'
  and o_orderdate < date '1994-01-01' + interval '1' year
group by
  n_name
order by
  revenue desc;
    """
    c = db.customer
    c = c[['c_custkey', 'c_nationkey']]
    o = db.orders
    l = db.lineitem
    s = db.supplier
    s = s[['s_nationkey', 's_suppkey']]
    n = db.nation
    n = n[['n_name', 'n_nationkey', 'n_regionkey']]
    r = db.region
    o = o[(o['o_orderdate'] >= datetime.date(1994, 1, 1))
         &(o['o_orderdate'] < datetime.date(1995, 1, 1))]
    o = o[['o_orderkey', 'o_custkey']]
    l = l[['l_suppkey', 'l_orderkey', 'l_discount', 'l_extendedprice']]
    r = r[r['r_name'] == 'ASIA']
    r = r[['r_regionkey']]

    t = c.merge(o, left_on='c_custkey', right_on='o_custkey')
    t = t.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t.merge(s, left_on=['l_suppkey', 'c_nationkey'], right_on=['s_suppkey', 's_nationkey'])
    t = t.merge(n, left_on='s_nationkey', right_on='n_nationkey')
    t = t.merge(r, left_on='n_regionkey', right_on='r_regionkey')
    t['revenue'] = t['l_extendedprice'] * (1 - t['l_discount'])
    t = t[['n_name', 'revenue']]
    t = t.groupby(('n_name'), sort=False).sum(min_count=1)
    t.reset_index(inplace=True)
    t.sort_values('revenue', ascending=False, inplace=True)

    return t;


def q06(db):
    """
select
  sum(l_extendedprice * l_discount) as revenue
from
  lineitem
where
  l_shipdate >= date '1994-01-01'
  and l_shipdate < date '1994-01-01' + interval '1' year
  and l_discount between .06 - 0.01 and .06 + 0.01
  and l_quantity < 24;
    """
    l = db.lineitem
    l = l[(l['l_shipdate'] >= datetime.date(1994, 1, 1))
         &(l['l_shipdate'] < datetime.date(1995, 1, 1))
         &(l['l_discount'] >= 0.05)
         &(l['l_discount'] <= 0.07)
         &(l['l_quantity'] < 24)]
    l['revenue'] = l['l_extendedprice'] * l['l_discount']
    l = l[['revenue']]
    l = l.sum(min_count=1)
    l = pd.DataFrame({'revenue': [l.revenue]})
    return l;


def q07(db):
    """
select
  supp_nation,
  cust_nation,
  l_year,
  sum(volume) as revenue
from
  (
    select
      n1.n_name as supp_nation,
      n2.n_name as cust_nation,
      extract(year from l_shipdate) as l_year,
      l_extendedprice * (1 - l_discount) as volume
    from
      supplier,
      lineitem,
      orders,
      customer,
      nation n1,
      nation n2
    where
      s_suppkey = l_suppkey
      and o_orderkey = l_orderkey
      and c_custkey = o_custkey
      and s_nationkey = n1.n_nationkey
      and c_nationkey = n2.n_nationkey
      and (
        (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
        or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
      )
      and l_shipdate between date '1995-01-01' and date '1996-12-31'
  ) as shipping
group by
  supp_nation,
  cust_nation,
  l_year
order by
  supp_nation,
  cust_nation,
  l_year;
    """
    s = db.supplier
    s = s[['s_suppkey', 's_nationkey']]
    l = db.lineitem
    l = l[(l['l_shipdate'] >= datetime.date(1995, 1, 1))
         &(l['l_shipdate'] <= datetime.date(1996, 12, 31))]
    l['revenue'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l['l_year'] = l['l_shipdate'].apply(lambda x: x.year)
    l = l[['l_suppkey', 'l_orderkey', 'l_year', 'revenue']]
    o = db.orders
    o = o[['o_custkey', 'o_orderkey']]
    c = db.customer
    c = c[['c_custkey', 'c_nationkey']]
    n = db.nation
    n = n[['n_name', 'n_nationkey']]

    ns = n.rename(columns={'n_name': 'supp_nation', 'n_nationkey': 'ns_nationkey'})
    nc = n.rename(columns={'n_name': 'cust_nation', 'n_nationkey': 'nc_nationkey'})

    t = s.merge(l, left_on='s_suppkey', right_on='l_suppkey')
    t = t.merge(o, left_on='l_orderkey', right_on='o_orderkey')
    t = t.merge(c, left_on='o_custkey', right_on='c_custkey')
    t = t.merge(ns, left_on='s_nationkey', right_on='ns_nationkey')
    t = t.merge(nc, left_on='c_nationkey', right_on='nc_nationkey')
    t = t[['supp_nation', 'cust_nation', 'l_year', 'revenue']]
    t = t[((t['supp_nation'] == 'FRANCE') & (t['cust_nation'] == 'GERMANY')) | ((t['supp_nation'] == 'GERMANY') & (t['cust_nation'] == 'FRANCE'))]
    t = t.groupby(('supp_nation', 'cust_nation', 'l_year')).sum(min_count=1)
    t.reset_index(inplace=True)
    t.sort_values(['supp_nation', 'cust_nation', 'l_year'])
    return t


def q08(db):
    """
select
  o_year,
  sum(case
    when nation = 'BRAZIL' then volume
    else 0
  end) / sum(volume) as mkt_share
from
  (
    select
      extract(year from o_orderdate) as o_year,
      l_extendedprice * (1 - l_discount) as volume,
      n2.n_name as nation
    from
      part,
      supplier,
      lineitem,
      orders,
      customer,
      nation n1,
      nation n2,
      region
    where
      p_partkey = l_partkey
      and s_suppkey = l_suppkey
      and l_orderkey = o_orderkey
      and o_custkey = c_custkey
      and c_nationkey = n1.n_nationkey
      and n1.n_regionkey = r_regionkey
      and r_name = 'AMERICA'
      and s_nationkey = n2.n_nationkey
      and o_orderdate between date '1995-01-01' and date '1996-12-31'
      and p_type = 'ECONOMY ANODIZED STEEL'
  ) as all_nations
group by
  o_year
order by
  o_year;
    """
    p = db.part
    p = p[p['p_type'] == 'ECONOMY ANODIZED STEEL']
    p = p[['p_partkey']]
    s = db.supplier
    s = s[['s_suppkey', 's_nationkey']]
    l = db.lineitem
    l = l[['l_partkey', 'l_suppkey', 'l_orderkey', 'l_extendedprice', 'l_discount']]
    o = db.orders
    o = o[(o['o_orderdate'] >= datetime.date(1995, 1, 1))
         &(o['o_orderdate'] < datetime.date(1996, 12, 31))]
    o = o[['o_custkey', 'o_orderkey', 'o_orderdate']]
    c = db.customer
    c = c[['c_custkey', 'c_nationkey']]
    n = db.nation
    n = n[['n_name', 'n_nationkey', 'n_regionkey']]
    r = db.region
    r = r[r['r_name'] == 'AMERICA']
    r = r[['r_regionkey']]

    n1 = n.rename(columns={'n_name': 'n1_name', 'n_nationkey': 'n1_nationkey', 'n_regionkey': 'n1_regionkey'})
    n2 = n.rename(columns={'n_name': 'n2_name', 'n_nationkey': 'n2_nationkey', 'n_regionkey': 'n2_regionkey'})

    t = l.merge(p, left_on='l_partkey', right_on='p_partkey')
    t = t.merge(s, left_on='l_suppkey', right_on='s_suppkey')
    t = t.merge(o, left_on='l_orderkey', right_on='o_orderkey')
    t = t.merge(c, left_on='o_custkey', right_on='c_custkey')
    t = t.merge(n1, left_on='c_nationkey', right_on='n1_nationkey')
    t = t.merge(r, left_on='n1_regionkey', right_on='r_regionkey')
    t = t.merge(n2, left_on='s_nationkey', right_on='n2_nationkey')

    t['o_year'] = t['o_orderdate'].apply(lambda x: x.year)
    t['volume2'] = t['l_extendedprice'] * (1 - t['l_discount'])
    t['volume1'] = t[['volume2', 'n2_name']].apply(lambda x: x[0] if x[1] == 'BRAZIL' else 0, axis=1)
    t = t[['o_year', 'volume1', 'volume2']]
    t = t.groupby('o_year').sum(min_count=1)
    t['mkt_share'] = t['volume1'] / t['volume2']
    t = t[['mkt_share']]
    t.reset_index(inplace=True)
    t.sort_values('o_year', inplace=True)
    return t


def q09(db):
    """
select
  nation,
  o_year,
  sum(amount) as sum_profit
from
  (
    select
      n_name as nation,
      extract(year from o_orderdate) as o_year,
      l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
    from
      part,
      supplier,
      lineitem,
      partsupp,
      orders,
      nation
    where
      s_suppkey = l_suppkey
      and ps_suppkey = l_suppkey
      and ps_partkey = l_partkey
      and p_partkey = l_partkey
      and o_orderkey = l_orderkey
      and s_nationkey = n_nationkey
      and p_name like '%green%'
  ) as profit
group by
  nation,
  o_year
order by
  nation,
  o_year desc;
    """
    p = db.part
    p = p[p['p_name'].str.contains('^.*green.*$')]
    p = p[['p_partkey']]
    s = db.supplier
    s = s[['s_suppkey', 's_nationkey']]
    l = db.lineitem
    l = l[['l_partkey', 'l_suppkey', 'l_orderkey', 'l_extendedprice', 'l_discount', 'l_quantity']]
    ps = db.partsupp
    ps = ps[['ps_suppkey', 'ps_partkey', 'ps_supplycost']]
    o = db.orders
    o = o[['o_orderkey', 'o_orderdate']]
    n = db.nation
    n = n[['n_name', 'n_nationkey']]

    t = s.merge(l, left_on='s_suppkey', right_on='l_suppkey')
    t = t.merge(ps, left_on=['l_suppkey', 'l_partkey'], right_on=['ps_suppkey', 'ps_partkey'])
    t = t.merge(p, left_on='l_partkey', right_on='p_partkey')
    t = t.merge(o, left_on='l_orderkey', right_on='o_orderkey')
    t = t.merge(n, left_on='s_nationkey', right_on='n_nationkey')
    t.rename(columns={'n_name': 'nation'}, inplace=True)
    t['o_year'] = t['o_orderdate'].apply(lambda x: x.year)
    t['sum_amount'] = t['l_extendedprice'] * (1 - t['l_discount']) - t['l_quantity'] * t['ps_supplycost']
    t = t[['nation', 'o_year', 'sum_amount']]
    t = t.groupby(('nation', 'o_year')).sum(min_count=1)
    t.reset_index(inplace=True)
    t.sort_values(['nation', 'o_year'], ascending=[True, False], inplace=True)
    return t


def q10(db):
    """
select
  c_custkey,
  c_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  c_acctbal,
  n_name,
  c_address,
  c_phone,
  c_comment
from
  customer,
  orders,
  lineitem,
  nation
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate >= date '1993-10-01'
  and o_orderdate < date '1993-10-01' + interval '3' month
  and l_returnflag = 'R'
  and c_nationkey = n_nationkey
group by
  c_custkey,
  c_name,
  c_acctbal,
  c_phone,
  n_name,
  c_address,
  c_comment
order by
  revenue desc
limit 20;
    """
    c = db.customer
    c = c[['c_custkey', 'c_nationkey', 'c_name', 'c_acctbal', 'c_address', 'c_phone', 'c_comment']]
    o = db.orders
    o = o[(o['o_orderdate'] >= datetime.date(1993, 10, 1))
         &(o['o_orderdate'] < datetime.date(1994, 1, 1))]
    o = o[['o_orderkey', 'o_custkey']]
    l = db.lineitem
    l = l[l['l_returnflag'] == 'R']
    l['revenue'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l = l[['l_orderkey', 'revenue']]
    n = db.nation
    n = n[['n_name', 'n_nationkey']]

    t = c.merge(o, left_on='c_custkey', right_on='o_custkey')
    t = t.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t.merge(n, left_on='c_nationkey', right_on='n_nationkey')
    t = t[['c_custkey', 'c_name', 'c_acctbal', 'c_phone', 'n_name', 'c_address', 'c_comment', 'revenue']]
    t = t.groupby(('c_custkey', 'c_name', 'c_acctbal', 'c_phone', 'n_name', 'c_address', 'c_comment')).sum(min_count=1)
    t.reset_index(inplace=True)
    t.sort_values('revenue', ascending=False, inplace=True)
    return t.head(20)


def q11(db):
    """
select
  ps_partkey,
  sum(ps_supplycost * ps_availqty) as value
from
  partsupp,
  supplier,
  nation
where
  ps_suppkey = s_suppkey
  and s_nationkey = n_nationkey
  and n_name = 'GERMANY'
group by
  ps_partkey having
    sum(ps_supplycost * ps_availqty) > (
      select
        sum(ps_supplycost * ps_availqty) * 0.0100000000
      --                                     ^^^^^^^^^^^^
      -- The above constant needs to be adjusted according
      -- to the scale factor (SF): constant = 0.0001 / SF.
      from
        partsupp,
        supplier,
        nation
      where
        ps_suppkey = s_suppkey
        and s_nationkey = n_nationkey
        and n_name = 'GERMANY'
    )
order by
  value desc;
    """
    ps = db.partsupp
    ps = ps[['ps_suppkey', 'ps_supplycost', 'ps_availqty', 'ps_partkey']]
    s = db.supplier
    s = s[['s_nationkey', 's_suppkey']]
    n = db.nation
    n = n[n['n_name'] == 'GERMANY']
    n = n[['n_name', 'n_nationkey']]

    j = ps.merge(s, left_on='ps_suppkey', right_on='s_suppkey')
    j = j.merge(n, left_on='s_nationkey', right_on='n_nationkey')

    ti = pd.DataFrame()
    ti['totsupcost'] = j['ps_supplycost'] * j['ps_availqty']
    ti = ti[['totsupcost']].sum(min_count=1) * 0.0001 / config.SF

    t = pd.DataFrame()
    t['ps_partkey'] = j['ps_partkey']
    t['value'] = j['ps_supplycost'] * j['ps_availqty']
    t = t[['ps_partkey', 'value']]
    t = t.groupby('ps_partkey').sum(min_count=1)
    t = t[t['value'] > ti[0]]
    t.reset_index(inplace=True)
    t.sort_values('value', ascending=False, inplace=True)
    return t


def q12(db):
    """
select
  l_shipmode,
  sum(case
    when o_orderpriority = '1-URGENT'
      or o_orderpriority = '2-HIGH'
      then 1
    else 0
  end) as high_line_count,
  sum(case
    when o_orderpriority <> '1-URGENT'
      and o_orderpriority <> '2-HIGH'
      then 1
    else 0
  end) as low_line_count
from
  orders,
  lineitem
where
  o_orderkey = l_orderkey
  and l_shipmode in ('MAIL', 'SHIP')
  and l_commitdate < l_receiptdate
  and l_shipdate < l_commitdate
  and l_receiptdate >= date '1994-01-01'
  and l_receiptdate < date '1994-01-01' + interval '1' year
group by
  l_shipmode
order by
  l_shipmode;
    """
    o = db.orders
    o = o[['o_orderkey', 'o_orderpriority']]
    l = db.lineitem
    l = l[(l['l_shipmode'].isin(['MAIL', 'SHIP']))
         &(l['l_commitdate'] < l['l_receiptdate'])
         &(l['l_shipdate'] < l['l_commitdate'])
         &(l['l_receiptdate'] >= datetime.date(1994, 1, 1))
         &(l['l_receiptdate'] < datetime.date(1995, 1, 1))]
    l = l[['l_orderkey', 'l_shipmode']]

    t = l.merge(o, left_on='l_orderkey', right_on='o_orderkey')
    def f(x):
        if x == '1-URGENT' or x == '2-HIGH':
            x1 = 1
        else:
            x1 = 0
        if x != '1-URGENT' and x != '2-HIGH':
            x2 = 1
        else:
            x2 = 0
        return x1, x2
    t['high_line_count'], t['low_line_count'] = zip(*t['o_orderpriority'].apply(f))
    t = t[['l_shipmode', 'high_line_count', 'low_line_count']]
    t = t.groupby('l_shipmode').sum(min_count=1)
    t.reset_index(inplace=True)
    t.sort_values('l_shipmode', inplace=True)
    return t


def q13(db):
    """
select
  c_count,
  count(*) as custdist
from
  (
    select
      c_custkey,
      count(o_orderkey)
    from
      customer left outer join orders on
        c_custkey = o_custkey
        and o_comment not like '%special%requests%'
    group by
      c_custkey
  ) as c_orders (c_custkey, c_count)
group by
  c_count
order by
  custdist desc,
  c_count desc;
    :param db:
    :return:
    """
    c = db.customer
    c = c[['c_custkey']]
    o = db.orders
    o = o[~o['o_comment'].str.contains('^.*special.*requests.*$')]
    o = o[['o_orderkey', 'o_custkey']]
    t = c.merge(o, left_on='c_custkey', right_on='o_custkey', how='left')
    t = t[['c_custkey', 'o_orderkey']]
    t = t.groupby('c_custkey').count()
    t.reset_index(inplace=True)
    t.rename(columns={'o_orderkey': 'c_count', 'c_custkey': 'custdist'}, inplace=True)
    t = t.groupby('c_count').count()
    t.reset_index(inplace=True)
    t.sort_values(['custdist', 'c_count'], ascending=[False, False], inplace=True)
    return t


def q14(db):
    """
select
  100.00 * sum(case
    when p_type like 'PROMO%'
      then l_extendedprice * (1 - l_discount)
    else 0
  end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
  lineitem,
  part
where
  l_partkey = p_partkey
  and l_shipdate >= date '1995-09-01'
  and l_shipdate < date '1995-09-01' + interval '1' month;
    """
    l = db.lineitem
    l = l[(l['l_shipdate'] >= datetime.date(1995, 9, 1))
         &(l['l_shipdate'] < datetime.date(1995, 10, 1))]
    l = l[['l_partkey', 'l_extendedprice', 'l_discount']]
    p = db.part
    p = p[['p_partkey', 'p_type']]

    t = l.merge(p, left_on='l_partkey', right_on='p_partkey')
    t['revenue2'] = t['l_extendedprice'] * (1 - t['l_discount'])
    t['revenue1'] = t['revenue2'].where(t['p_type'].str.contains('^PROMO.*$'), 0)
    t = t.sum(min_count=1)
    t = 100 * t['revenue1'] / t['revenue2']
    t = pd.DataFrame({'promo_revenue': [t]})
    return t;


def q15(db):
    """
create view revenue0 (supplier_no, total_revenue) as
  select
    l_suppkey,
    sum(l_extendedprice * (1 - l_discount))
  from
    lineitem
  where
    l_shipdate >= date '1996-01-01'
    and l_shipdate < date '1996-01-01' + interval '3' month
  group by
    l_suppkey;

select
  s_suppkey,
  s_name,
  s_address,
  s_phone,
  total_revenue
from
  supplier,
  revenue0
where
  s_suppkey = supplier_no
  and total_revenue = (
    select
      max(total_revenue)
    from
      revenue0
  )
order by
  s_suppkey;
drop view revenue0;
    """
    s = db.supplier
    s = s[['s_suppkey', 's_name', 's_address', 's_phone']]
    l = db.lineitem
    l = l[(l['l_shipdate'] >= datetime.date(1996, 1, 1))
         &(l['l_shipdate'] < datetime.date(1996, 4, 1))]
    l['total_revenue'] = l['l_extendedprice'] * (1 - l['l_discount'])
    l = l[['l_suppkey', 'total_revenue']]
    l = l.groupby('l_suppkey').sum(min_count=1)
    l.reset_index(inplace=True)

    ti = l.agg(max)

    t = s.merge(l, left_on='s_suppkey', right_on='l_suppkey')
    t = t[t['total_revenue'] == ti['total_revenue']]
    t = t[['s_suppkey', 's_name', 's_address', 's_phone', 'total_revenue']]
    t.sort_values('s_suppkey', inplace=True)
    return t


def q16(db):
    """
select
  p_brand,
  p_type,
  p_size,
  count(distinct ps_suppkey) as supplier_cnt
from
  partsupp,
  part
where
  p_partkey = ps_partkey
  and p_brand <> 'Brand#45'
  and p_type not like 'MEDIUM POLISHED%'
  and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
  and ps_suppkey not in (
    select
      s_suppkey
    from
      supplier
    where
      s_comment like '%Customer%Complaints%'
  )
group by
  p_brand,
  p_type,
  p_size
order by
  supplier_cnt desc,
  p_brand,
  p_type,
  p_size;
    """
    s = db.supplier
    s = s[s['s_comment'].str.contains('^.*Customer.*Complaints.*$')]
    ps = db.partsupp
    ps = ps[~ps['ps_suppkey'].isin(s['s_suppkey'])]
    ps = ps[['ps_partkey', 'ps_suppkey']]
    p = db.part
    p = p[(p['p_brand'] != 'Brand#45')
         &~(p['p_type'].str.contains('^MEDIUM POLISHED.*$'))
         &(p['p_size'].isin([49, 14, 23, 45, 19, 3, 36, 9]))]
    p = p[['p_partkey', 'p_brand', 'p_type', 'p_size']]

    t = p.merge(ps, left_on='p_partkey', right_on='ps_partkey')
    t = t[['p_brand', 'p_type', 'p_size', 'ps_suppkey']]
    t = t.groupby(('p_brand', 'p_type', 'p_size')).ps_suppkey.nunique()
    t.rename('supplier_cnt', inplace=True)
    t = t.reset_index()
    t.sort_values(['supplier_cnt', 'p_brand', 'p_type', 'p_size'], ascending=[False, True, True, True], inplace=True)
    return t


def q17(db):
    """
select
  sum(l_extendedprice) / 7.0 as avg_yearly
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and p_brand = 'Brand#23'
  and p_container = 'MED BOX'
  and l_quantity < (
    select
      0.2 * avg(l_quantity)
    from
      lineitem
    where
      l_partkey = p_partkey
  );

-- Rewritten to avoid using a corelated query

select
  sum(l_extendedprice) / 7.0 as avg_yearly
from
  lineitem,
  part,
(
    select p_partkey as i_partkey, 0.2 * avg(l_quantity) as  avg_qty
    from
      lineitem,
			part
    where
      l_partkey = p_partkey
	group by p_partkey
)x
where
  p_partkey = l_partkey
  and p_brand = 'Brand#23'
  and p_container = 'MED BOX'
	and p_partkey = i_partkey
  and l_quantity < avg_qty;
    """
    l = db.lineitem
    l = l[['l_partkey', 'l_quantity', 'l_extendedprice']]
    p = db.part
    p = p[['p_partkey', 'p_brand', 'p_container']]

    ti = l.merge(p, left_on='l_partkey', right_on='p_partkey')
    t = ti
    ti = ti.rename(columns={'p_partkey': 'i_partkey'})
    ti = ti[['i_partkey', 'l_quantity']]
    ti = ti.groupby('i_partkey').agg('mean')
    ti['avg_qty'] = ti['l_quantity'] * 0.2
    ti.reset_index(inplace=True)
    ti = ti[['i_partkey', 'avg_qty']]

    t = t[(t['p_brand'] == 'Brand#23') & (t['p_container'] == 'MED BOX')]
    t = t.merge(ti, left_on='p_partkey', right_on='i_partkey')
    t = t[t['l_quantity'] < t['avg_qty']]
    t = t[['l_extendedprice']]
    t = t.sum(min_count=1)
    t = pd.DataFrame({'avg_yearly': [t.l_extendedprice / 7.0]})
    return t


def q18(db):
    """
select
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice,
  sum(l_quantity)
from
  customer,
  orders,
  lineitem
where
  o_orderkey in (
    select
      l_orderkey
    from
      lineitem
    group by
      l_orderkey having
        sum(l_quantity) > 300
  )
  and c_custkey = o_custkey
  and o_orderkey = l_orderkey
group by
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice
order by
  o_totalprice desc,
  o_orderdate
limit 100;
    """
    c = db.customer
    c = c[['c_custkey', 'c_name']]
    o = db.orders
    o = o[['o_orderkey', 'o_orderdate', 'o_totalprice', 'o_custkey']]
    l = db.lineitem
    l = l[['l_orderkey', 'l_quantity']]

    ti = l[['l_orderkey', 'l_quantity']]
    ti = ti.groupby('l_orderkey').sum(min_count=1)
    ti = ti[ti['l_quantity'] > 300]
    ti.reset_index(inplace=True)

    t = c.merge(o, left_on='c_custkey', right_on='o_custkey')
    t = t.merge(l, left_on='o_orderkey', right_on='l_orderkey')
    t = t[t['o_orderkey'].isin(ti['l_orderkey'])]
    t = t[['c_name', 'c_custkey', 'o_orderkey', 'o_orderdate', 'o_totalprice', 'l_quantity']]
    t = t.groupby(('c_name', 'c_custkey', 'o_orderkey', 'o_orderdate', 'o_totalprice')).sum(min_count=1)
    t.reset_index(inplace=True)
    t.sort_values(['o_totalprice', 'o_orderdate'], ascending=[False, True], inplace=True)
    return t.head(100)


def q19(db):
    """
select
  sum(l_extendedprice* (1 - l_discount)) as revenue
from
  lineitem,
  part
where
  (
    p_partkey = l_partkey
    and p_brand = 'Brand#12'
    and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    and l_quantity >= 1 and l_quantity <= 1 + 10
    and p_size between 1 and 5
    and l_shipmode in ('AIR', 'AIR REG')
    and l_shipinstruct = 'DELIVER IN PERSON'
  )
  or
  (
    p_partkey = l_partkey
    and p_brand = 'Brand#23'
    and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
    and l_quantity >= 10 and l_quantity <= 10 + 10
    and p_size between 1 and 10
    and l_shipmode in ('AIR', 'AIR REG')
    and l_shipinstruct = 'DELIVER IN PERSON'
  )
  or
  (
    p_partkey = l_partkey
    and p_brand = 'Brand#34'
    and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
    and l_quantity >= 20 and l_quantity <= 20 + 10
    and p_size between 1 and 15
    and l_shipmode in ('AIR', 'AIR REG')
    and l_shipinstruct = 'DELIVER IN PERSON'
  );
    """
    l = db.lineitem
    l = l[(l['l_shipmode'].isin(['AIR', 'AIR REG']))
         &(l['l_shipinstruct'] == 'DELIVER IN PERSON')]
    l = l[['l_quantity', 'l_extendedprice', 'l_discount', 'l_partkey']]
    p = db.part
    p = p[['p_partkey', 'p_brand', 'p_container', 'p_size']]
    t = l.merge(p, left_on='l_partkey', right_on='p_partkey')
    t = t[ (  (t['p_brand'] == 'Brand#12')
            & (t['p_container'].isin(['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']))
            & (t['l_quantity'] >= 1) & (t['l_quantity'] <= 11)
            & (t['p_size'] >= 1) & (t['p_size'] <= 5)
        ) |(  (t['p_brand'] == 'Brand#23')
            & (t['p_container'].isin(['MED BAG', 'MED BOX', 'MED PKG', 'MED PACK']))
            & (t['l_quantity'] >= 10) & (t['l_quantity'] <= 20)
            & (t['p_size'] >= 1) & (t['p_size'] <= 10)
        ) |(  (t['p_brand'] == 'Brand#34')
            & (t['p_container'].isin(['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG']))
            & (t['l_quantity'] >= 20) & (t['l_quantity'] <= 30)
            & (t['p_size'] >= 1) & (t['p_size'] <= 15)
        )]
    t['revenue'] = t['l_extendedprice'] * (1 - t['l_discount'])
    t = t[['revenue']].sum(min_count=1)
    t = pd.DataFrame({'revenue': [t.revenue]})
    return t


def q20(db):
    """
select
  s_name,
  s_address
from
  supplier,
  nation
where
  s_suppkey in (
    select
      ps_suppkey
    from
      partsupp
    where
      ps_partkey in (
        select
          p_partkey
        from
          part
        where
          p_name like 'forest%'
      )
      and ps_availqty > (
        select
          0.5 * sum(l_quantity)
        from
          lineitem
        where
          l_partkey = ps_partkey
          and l_suppkey = ps_suppkey
          and l_shipdate >= date '1994-01-01'
          and l_shipdate < date '1994-01-01' + interval '1' year
      )
  )
  and s_nationkey = n_nationkey
  and n_name = 'CANADA'
order by
  s_name;

-- rewritten to avoid corelated query

select
  s_name,
  s_address
from
  supplier,
  nation
where
  s_suppkey in
	(
		select
			ps_suppkey
		from
			partsupp,
      (
        select
					l_partkey,
					l_suppkey,
          0.5 * sum(l_quantity) as totqty
        from
          lineitem
        where
          l_shipdate >= date '1994-01-01'
          and l_shipdate < date '1994-01-01' + interval '1' year
        group by
					l_partkey,
					l_suppkey
      )l
		where ps_partkey = l_partkey
			and ps_suppkey = l_suppkey
			and ps_availqty > totqty
			and ps_partkey in
			(
        select
          p_partkey
        from
          part
        where
          p_name like 'forest%'
      )
  )
  and s_nationkey = n_nationkey
  and n_name = 'CANADA'
order by
  s_name;
    """
    s = db.supplier
    s = s[['s_nationkey', 's_suppkey', 's_name', 's_address']]
    n = db.nation
    n = n[n['n_name'] == 'CANADA']
    n = n[['n_nationkey']]
    p = db.part
    p = p[p['p_name'].str.contains('^forest.*$')]
    l = db.lineitem
    l = l[(l['l_shipdate'] >= datetime.date(1994, 1, 1))
         &(l['l_shipdate'] < datetime.date(1995, 1, 1))]
    l = l[['l_partkey', 'l_suppkey', 'l_quantity']].groupby(('l_partkey', 'l_suppkey')).sum()
    l['totqty'] = l['l_quantity'] * 0.5
    l.reset_index(inplace=True)
    l = l[['l_partkey', 'l_suppkey', 'totqty']]
    ps = db.partsupp
    ps = ps[['ps_partkey', 'ps_suppkey', 'ps_availqty']]

    ti = ps.merge(l, left_on=['ps_partkey', 'ps_suppkey'], right_on=['l_partkey', 'l_suppkey'])
    ti = ti[(ti['ps_availqty'] > ti['totqty']) & (ti['ps_partkey'].isin(p.p_partkey))]

    t = s.merge(n, left_on='s_nationkey', right_on='n_nationkey')
    t = t[t['s_suppkey'].isin(ti.ps_suppkey)]
    t = t[['s_name', 's_address']]
    t.sort_values('s_name', inplace=True)
    return t


def q21(db):
    """
select
  s_name,
  count(*) as numwait
from
  supplier,
  lineitem l1,
  orders,
  nation
where
  s_suppkey = l1.l_suppkey
  and o_orderkey = l1.l_orderkey
  and o_orderstatus = 'F'
  and l1.l_receiptdate > l1.l_commitdate
  and exists (
    select
      *
    from
      lineitem l2
    where
      l2.l_orderkey = l1.l_orderkey
      and l2.l_suppkey <> l1.l_suppkey
  )
  and not exists (
    select
      *
    from
      lineitem l3
    where
      l3.l_orderkey = l1.l_orderkey
      and l3.l_suppkey <> l1.l_suppkey
      and l3.l_receiptdate > l3.l_commitdate
  )
  and s_nationkey = n_nationkey
  and n_name = 'SAUDI ARABIA'
group by
  s_name
order by
  numwait desc,
  s_name
limit 100;

-- rewritten to avoid corelated query

select
  s_name,
  count(*) as numwait
from
  supplier,
  lineitem l1,
  orders,
  nation
where
  s_suppkey = l1.l_suppkey
  and o_orderkey = l1.l_orderkey
  and o_orderstatus = 'F'
  and l1.l_receiptdate > l1.l_commitdate
	and (l1.l_orderkey, l1.l_suppkey) in
	(
	  select t1.l_orderkey, t1.l_suppkey
		from lineitem t1, lineitem t2
		where t1.l_orderkey = t2.l_orderkey
		  and t1.l_suppkey <> t2.l_suppkey
		--group by t1.l_orderkey, t1.l_suppkey
	)
	and (l1.l_orderkey, l1.l_suppkey) not in
	(
	  select t1.l_orderkey, t1.l_suppkey
		from lineitem t1, lineitem t2
		where t1.l_orderkey = t2.l_orderkey
		  and t1.l_suppkey <> t2.l_suppkey
			and t2.l_receiptdate > t2.l_commitdate
		--group by t1.l_orderkey, t1.l_suppkey
	)
  and s_nationkey = n_nationkey
  and n_name = 'SAUDI ARABIA'
group by
  s_name
order by
  numwait desc,
  s_name
limit 100;
    """
    s = db.supplier
    s = s[['s_suppkey', 's_nationkey', 's_name']]
    l = db.lineitem
    o = db.orders
    o = o[o['o_orderstatus'] == 'F']
    o = o[['o_orderkey']]
    n = db.nation
    n = n[n['n_name'] == 'SAUDI ARABIA']
    n = n[['n_nationkey']]

    l1 = l[['l_orderkey', 'l_suppkey']].rename(columns={'l_orderkey': 'l1_orderkey', 'l_suppkey': 'l1_suppkey'})
    l = l[l['l_receiptdate'] > l['l_commitdate']]
    l2 = l[['l_orderkey', 'l_suppkey']].rename(columns={'l_orderkey': 'l2_orderkey', 'l_suppkey': 'l2_suppkey'})
    l = l[['l_orderkey', 'l_suppkey']]
    t = s.merge(l, left_on='s_suppkey', right_on='l_suppkey')
    t = t.merge(o, left_on='l_orderkey', right_on='o_orderkey')
    t = t.merge(n, left_on='s_nationkey', right_on='n_nationkey')
    t = t[['s_name', 'l_orderkey', 'l_suppkey']]

    t1 = t.merge(l1, left_on='l_orderkey', right_on='l1_orderkey')
    t1 = t1[t1['l_suppkey'] != t1['l1_suppkey']][['l_orderkey', 'l_suppkey']].rename(columns={'l_orderkey': 'l1_orderkey', 'l_suppkey': 'l1_suppkey'})
    t = t.merge(t1, left_on=['l_orderkey', 'l_suppkey'], right_on=['l1_orderkey', 'l1_suppkey'])
    t2 = t.merge(l2, left_on='l_orderkey', right_on='l2_orderkey')
    t2 = t2[t2['l_suppkey'] != t2['l2_suppkey']][['l_orderkey', 'l_suppkey']].rename(columns={'l_orderkey': 'l2_orderkey', 'l_suppkey': 'l2_suppkey'})
    t = t.merge(t2, left_on=['l_orderkey', 'l_suppkey'], right_on=['l2_orderkey', 'l2_suppkey'], how='left', indicator='Exist')
    print(t)
    t = t[t['Exist'] == 'left_only']
    t = t[['s_name', 'l_orderkey']].rename(columns={'l_orderkey': 'numwait'})
    t = t.groupby('s_name').nunique()[['numwait']]
    t.reset_index(inplace=True)
    t.sort_values(['numwait', 's_name'], ascending=[False, True], inplace=True)
    return t.head(100)


def q22(db):
    """
select
  cntrycode,
  count(*) as numcust,
  sum(c_acctbal) as totacctbal
from
  (
    select
      substring(c_phone from 1 for 2) as cntrycode,
      c_acctbal
    from
      customer
    where
      substring(c_phone from 1 for 2) in
        ('13', '31', '23', '29', '30', '18', '17')
      and c_acctbal > (
        select
          avg(c_acctbal)
        from
          customer
        where
          c_acctbal > 0.00
          and substring(c_phone from 1 for 2) in
            ('13', '31', '23', '29', '30', '18', '17')
      )
      and not exists (
        select
          *
        from
          orders
        where
          o_custkey = c_custkey
      )
  ) as custsale
group by
  cntrycode
order by
  cntrycode;

-- rewritten to avoid corelated query

select
  cntrycode,
  count(*) as numcust,
  sum(c_acctbal) as totacctbal
from
  (
    select
      substring(c_phone from 1 for 2) as cntrycode,
      c_acctbal
    from
      customer
    where
      substring(c_phone from 1 for 2) in
        ('13', '31', '23', '29', '30', '18', '17')
      and c_acctbal > (
        select
          avg(c_acctbal)
        from
          customer
        where
          c_acctbal > 0.00
          and substring(c_phone from 1 for 2) in
            ('13', '31', '23', '29', '30', '18', '17')
      )
      and c_custkey not in (
        select
          o_custkey
        from
          orders
      )
  ) as custsale
group by
  cntrycode
order by
  cntrycode;
    """
    o = db.orders
    o = o[['o_custkey']]
    c = db.customer
    c['cntrycode'] = c['c_phone'].apply(lambda x: x[0:2])
    c = c[c['cntrycode'].isin(['13', '31', '23', '29', '30', '18', '17'])]
    c = c[['cntrycode', 'c_acctbal', 'c_custkey']]

    ti = c[c['c_acctbal'] > 0.0]
    ti = ti[['c_acctbal']].agg('mean')
    t = c[(c['c_acctbal'] > ti.c_acctbal) & ~(c['c_custkey'].isin(o.o_custkey))]
    t = t[['cntrycode', 'c_acctbal']].groupby('cntrycode').agg(['count', 'sum'])
    t.columns = pd.Index([t.columns.levels[1][t.columns.labels[1][i]] + '_' + t.columns.levels[0][t.columns.labels[0][i]] for i in range(len(t.columns.labels[0]))])
    t.rename(columns={'count_c_acctbal': 'numcust', 'sum_c_acctbal': 'totacctbal'}, inplace=True)
    t.reset_index(inplace=True)
    t.sort_values('cntrycode', inplace=True)
    return t
