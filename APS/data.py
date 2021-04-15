from dataclasses import dataclass

@dataclass
class Data:
    img_id: str
    obj_name: str
    pbbox: []
    obbox: []



def proc(data:Data):
    print(data.img_id)
    print(data.obj_name)
    print(data.pbbox)
    print(data.obbox)

if __name__ == '__main__':

    print(proc(Data('test', 'tt', [1,2,3,4], [1,2,3,4])))
    print(proc(Data('test', 'tt', [2,1,5,8], [1,2,3,4])))
    print(proc(Data('test', 'tt', [1,2,3,4], [1,2,3,4])))
    print(proc(Data('test', 'tt', [1,2,3,4], [5,20,5,6])))
