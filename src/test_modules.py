from category import *


class TestFileModule(object):
    _loc = ['ca', 'de', 'fr', 'gb', 'us']
    _csv = ['../datasets/CAvideos.csv',
            '../datasets/DEvideos.csv',
            '../datasets/FRvideos.csv',
            '../datasets/GBvideos.csv',
            '../datasets/USvideos.csv']
    _json = ['../datasets/CA_category_id.json',
             '../datasets/DE_category_id.json',
             '../datasets/FR_category_id.json',
             '../datasets/GB_category_id.json',
             '../datasets/US_category_id.json']

    def test_all_csv(self):
        ans = TestFileModule._csv
        test = [i for i in all_csv()]
        test.sort()
        assert len(test) == len(ans)
        assert all(test[i] == ans[i] for i in range(len(test)))

    def test_all_json(self):
        ans = TestFileModule._json
        test = [i for i in all_json()]
        test.sort()
        assert len(test) == len(ans)
        assert all(test[i] == ans[i] for i in range(len(test)))

    def test_csv_at(self):
        loc = TestFileModule._loc
        ans = TestFileModule._csv
        assert all(ans[i] == csv_at(v) for i,v in enumerate(loc))

    def test_json_at(self):
        loc = TestFileModule._loc
        ans = TestFileModule._json
        assert all(ans[i] == json_at(v) for i,v in enumerate(loc))

