import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.chrome.ChromeOptions;

import static com.codeborne.selenide.Configuration.baseUrl;
import static com.codeborne.selenide.Configuration.browserCapabilities;
import static com.codeborne.selenide.Selenide.open;
import static com.codeborne.selenide.Selenide.page;


public class CreditTests {

    private final ActionSteps action = new ActionSteps();

    @BeforeAll
    public static void setUp() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--incognito");
        browserCapabilities.setCapability(ChromeOptions.CAPABILITY, options);
        baseUrl = "https://alfabank.ru/get-money/credit/credit-cash/step1/";
        System.setProperty("webdriver.chrome.driver", "src/test/resources/chromedriver.exe");
    }

    @Test
    public void checkCorrectInput() {
        open(baseUrl);
        action.setValueInField(page(CreditPage.class).fullNameField, "Глебов Глеб Глебович");
        action.setValueInField(page(CreditPage.class).emailField, "test@gmail.com");
        action.setValueInField(page(CreditPage.class).phoneField, "9342225543");
    }

    @Test
    public void checkFullNameWithoutSpacesInput() {
        open(baseUrl);
        action.setValueInField(page(CreditPage.class).fullNameField, "крокозябра");
        action.selectCheckbox(page(CreditPage.class).agreeToReceiveAdsCheckbox);
//        action.selectCheckbox(page(CreditPage.class).withoutMiddleNameCheckbox);
        action.checkText(page(CreditPage.class).fioWithoutSpacesWarning, "Проверьте и заполните недостающие поля");
    }

    @Test
    public void checkSelectOption() {
        open(baseUrl);
        action.selectCheckbox(page(CreditPage.class).agreeToReceiveAdsCheckbox);
        action.selectCheckbox(page(CreditPage.class).conditionsCheckbox);
    }

    @Test
    public void checkSuccessApplyForLoan() {
        open(baseUrl);
        action.setValueInField(page(CreditPage.class).fullNameField, "Глебов Глеб Глебович");
        action.setValueInField(page(CreditPage.class).emailField, "test@gmail.com");
        action.setValueInField(page(CreditPage.class).phoneField, "9342225543");
        action.selectCheckbox(page(CreditPage.class).agreeToReceiveAdsCheckbox);
        action.selectCheckbox(page(CreditPage.class).conditionsCheckbox);
        //нужно дописать нажатие на кнопку выбора пола
        action.clickBtn(page(CreditPage.class).submitBtn);
        action.checkText(page(CreditPage.class).wayToStepTwo, "Заявка на кредит наличными или рефинансирование");
    }


}
