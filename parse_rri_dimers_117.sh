for i in `ls dimers_117/computedFeatures/common/contactMaps/  |  grep cMap  | cut -c1-4  | sort  | uniq`; do
  echo $i
  awk  '{if($7==1 && $2!~/[A-Z]/ && $5!~/[A-Z]/) print $5$4"\t"$2$1}'  dimers_117/computedFeatures/common/contactMaps/$i.cMap.tab > dimers_117_rri/$i.int
done
