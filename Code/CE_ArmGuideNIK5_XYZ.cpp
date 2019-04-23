#include <Arm7Bot.h>
#include<string.h>
#include<stdio.h>
#include<stdlib.h>
Arm7Bot Arm;
#define SERVO_NUM 7       //可控自由度数 
double offsetValues[SERVO_NUM] = {-9,2,-2,-10,0,2,0};
//xy坐标————各自由度位置
//x-y坐标结构体
struct return_pxy
{
	double int_px;
	double int_py;
    double int_pz;
};

void setup()
{
    Arm.begin(&Serial);
    Arm.initialMove();
	Serial.begin(9600);
}

void loop()
{
    //int done_p_x; int done_p_y;
    struct return_pxy stu_position;
    char receive_char;
    char serial_receive[30];
    int select_num,read_flag=0,bit_temp,read_open=1;
    while(1){
        if(Serial.available()){
        receive_char = Serial.read();
        if(receive_char == 'F')
        {
            read_flag = 1;
            bit_temp = 0;
        }
        if(read_flag)
        {
            if(receive_char == 'L')
            {
                serial_receive[bit_temp-1] = '\0';
                read_flag = 0;
                break;
            }
            else
            {
                if(bit_temp){
                serial_receive[bit_temp-1] = receive_char;
                bit_temp++;
                }
                else bit_temp++;
            }    
        }
    }
    }
    stu_position=get_position(serial_receive);
    step_catch(stu_position.int_px,stu_position.int_py,stu_position.int_pz);
    Serial.print("1");
  
}








//抓取操作
void step_catch(double a,double b,double c)
{
  for(int i=0; i<SERVO_NUM; i++)
  Arm.offset[i] = offsetValues[i];
  while (!Arm.allConverge()) delay(20);
  Arm.moveIK5( {a, b, 100}, {0, 0, -1}, 60, 50 );
  while (!Arm.allConverge()) delay(20);
  delay(1000);

  Arm.moveIK5( {a, b, c}, {0, 0, -1}, 60, 10 );
  while (!Arm.allConverge()) delay(20);
  delay(1000);

  Arm.moveIK5( {-180, 200, 100}, {0, 0,-1}, 60, 10 );
  while (!Arm.allConverge()) delay(20);
  delay(1000);

  Arm.moveIK5( {-180, 245, 35}, {0, 0, -1}, 60, 10 );
  while (!Arm.allConverge()) delay(20);
  delay(1000);

  Arm.moveIK5( {-180, 245, 35}, {0, 0, -1}, 60, 50 );
  while (!Arm.allConverge()) delay(20);
  delay(1000);
  
  Arm.moveIK5( {-180, 200, 80}, {0, 0, -1}, 60, 50 );
  while (!Arm.allConverge()) delay(20);
  delay(1000);
  
  Arm.initialMove();
  while (!Arm.allConverge()) delay(20);
}


//读取串口字符串(char)并进行分割、转为x，y坐标(int)
//返回x-y坐标结构体
struct return_pxy get_position(char serial_input[])
{
	struct return_pxy stu_position;
    int i;
    char*temp;
    double receive_xyz[4];
    double p_x,p_y,p_z;
    temp = strtok(serial_input,"/"); 
    receive_xyz[0] = atoi(temp);
    for(i=1;i<5;i++)
    {
         receive_xyz[i] = 0;
         temp = strtok(NULL, "/");
		 receive_xyz[i] = atoi(temp);
    }
    stu_position.int_px = receive_xyz[0];
    stu_position.int_py = receive_xyz[1];
    stu_position.int_pz = receive_xyz[2];
	return stu_position;
}

