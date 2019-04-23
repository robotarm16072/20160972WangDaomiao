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
    step_catch(stu_position.int_px,stu_position.int_py);
    Serial.print("1");
  
}








//抓取操作
void step_catch(double a,double b)
{
  for(int i=0; i<SERVO_NUM; i++)
  Arm.offset[i] = offsetValues[i];
  while (!Arm.allConverge()) delay(20);

  Arm.moveIK5( {a, b, 35}, {0, 0, -1}, 60, 10 );
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
    char*p_x; char*p_y;
    double int_p_x,int_p_y;
    p_x = strtok(serial_input, "/");
    int_p_x = atoi(p_x);
	stu_position.int_px = int_p_x ;
    
    p_y = strtok(NULL, "/");
    int_p_y = atoi(p_y);
	stu_position.int_py =  int_p_y;
	return stu_position;
}

