import com.codeborne.selenide.SelenideElement;
import org.openqa.selenium.support.FindBy;

public class CreditPage {

    @FindBy(xpath = "//input[@name='fullName']")
    public SelenideElement fullNameField;

    @FindBy(xpath = "//input[@data-test-id='email-input']")
    public SelenideElement emailField;

    @FindBy(xpath = "//input[@data-test-id='phoneInput']")
    public SelenideElement phoneField;

    @FindBy(xpath = "//input[@data-test-id='phoneInput']")
    public SelenideElement conditionsCheckbox;

    @FindBy(xpath = "//div[@data-test-id='isAdvertisingAccepted-checkbox-caption']")
    public SelenideElement agreeToReceiveAdsCheckbox;

    @FindBy(xpath = "//button[@data-test-id='submit-button']")
    public SelenideElement submitBtn;

    @FindBy(xpath = "//input[@data-test-id='hasMiddleName-checkbox']")
    public SelenideElement withoutMiddleNameCheckbox;

    @FindBy(xpath = "//h2[@data-test-id='header-title']")
    public SelenideElement wayToStepTwo;

    @FindBy(xpath = "//span[@data-test-id='fio-data-warning']")
    public SelenideElement fioWithoutSpacesWarning;

    @FindBy(xpath = "//div[@data-widget-name='PilSecurityWarrantyBlock']")
    public SelenideElement spaceToClick;

}

